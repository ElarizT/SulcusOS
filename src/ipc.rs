use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use tokio::runtime::{Builder, Handle};
use tokio::sync::{mpsc, oneshot};

type MailboxReceiver = Arc<tokio::sync::Mutex<mpsc::Receiver<AgentMessage>>>;

#[pyclass]
#[derive(Clone, Debug)]
pub struct AgentMessage {
    #[pyo3(get, set)]
    pub sender: String,
    #[pyo3(get, set)]
    pub receiver: String,
    #[pyo3(get, set)]
    pub payload: String,
}

#[pymethods]
impl AgentMessage {
    #[new]
    pub fn new(sender: String, receiver: String, payload: String) -> PyResult<Self> {
        validate_json_payload(&payload)?;

        Ok(Self {
            sender,
            receiver,
            payload,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentMessage(sender={:?}, receiver={:?}, payload={:?})",
            self.sender, self.receiver, self.payload
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct RustKernel {
    inner: Arc<KernelInner>,
}

#[pymethods]
impl RustKernel {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_agent_capability(
        &self,
        agent_name: String,
        capability: String,
    ) -> PyResult<()> {
        self.inner
            .register_agent_capability(agent_name, capability)
    }

    pub fn unregister_agent(&self, agent_name: String) -> PyResult<()> {
        self.inner.unregister_agent(&agent_name)
    }

    pub fn find_agents_by_capability(&self, capability: String) -> PyResult<Vec<String>> {
        self.inner.find_agents_by_capability(&capability)
    }

    pub fn total_registered_agents(&self) -> PyResult<usize> {
        self.inner.total_registered_agents()
    }

    pub fn shutdown(&self) -> PyResult<()> {
        let mut is_shutting_down = self.inner.is_shutting_down.lock().map_err(lock_error)?;
        *is_shutting_down = true;
        Ok(())
    }

    pub fn is_shutting_down(&self) -> PyResult<bool> {
        let is_shutting_down = self.inner.is_shutting_down.lock().map_err(lock_error)?;
        Ok(*is_shutting_down)
    }
}

impl Default for RustKernel {
    fn default() -> Self {
        Self {
            inner: Arc::new(KernelInner::default()),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct NativeIPCBus {
    inner: Arc<BusInner>,
}

#[pymethods]
impl NativeIPCBus {
    #[new]
    #[pyo3(signature = (kernel=None))]
    pub fn new(kernel: Option<PyRef<'_, RustKernel>>) -> PyResult<Self> {
        let kernel = match kernel {
            Some(kernel) => Arc::clone(&kernel.inner),
            None => Arc::new(KernelInner::default()),
        };

        Ok(Self {
            inner: Arc::new(BusInner::start(kernel)?),
        })
    }

    pub fn register_mailbox(&self, agent_name: String, buffer_size: usize) -> PyResult<()> {
        if buffer_size == 0 {
            return Err(PyValueError::new_err("buffer_size must be greater than zero"));
        }

        let (sender, receiver) = mpsc::channel(buffer_size);

        let mut senders = self.inner.senders.lock().map_err(lock_error)?;
        let mut receivers = self.inner.receivers.lock().map_err(lock_error)?;

        if senders.contains_key(&agent_name) {
            return Err(PyValueError::new_err(format!(
                "mailbox for agent '{agent_name}' is already registered"
            )));
        }

        senders.insert(agent_name.clone(), sender);
        receivers.insert(
            agent_name.clone(),
            Arc::new(tokio::sync::Mutex::new(receiver)),
        );
        self.inner
            .routing_methods
            .lock()
            .map_err(lock_error)?
            .insert(agent_name.clone(), "Direct".to_string());
        self.inner.kernel.register_agent(agent_name)?;

        Ok(())
    }

    pub fn unregister_mailbox(&self, agent_name: String) -> PyResult<bool> {
        let mut senders = self.inner.senders.lock().map_err(lock_error)?;
        let mut receivers = self.inner.receivers.lock().map_err(lock_error)?;
        let mut routing_methods = self.inner.routing_methods.lock().map_err(lock_error)?;

        let removed = senders.remove(&agent_name).is_some();
        receivers.remove(&agent_name);
        routing_methods.remove(&agent_name);
        self.inner.kernel.unregister_agent(&agent_name)?;

        Ok(removed)
    }

    pub fn send_message(&self, msg: PyRef<'_, AgentMessage>) -> PyResult<()> {
        let mut msg = msg.clone();

        validate_json_payload(&msg.payload)?;

        let (target_agent, sender, routing_method) = self.resolve_delivery(&msg.receiver)?;
        msg.receiver = target_agent;
        self.inner
            .routing_methods
            .lock()
            .map_err(lock_error)?
            .insert(msg.receiver.clone(), routing_method);

        sender.try_send(msg).map_err(|err| match err {
            mpsc::error::TrySendError::Full(msg) => PyRuntimeError::new_err(format!(
                "mailbox for receiver '{}' is full",
                msg.receiver
            )),
            mpsc::error::TrySendError::Closed(msg) => PyRuntimeError::new_err(format!(
                "mailbox for receiver '{}' is closed",
                msg.receiver
            )),
        })
    }

    pub fn recv_message(&self, py: Python<'_>, agent_name: String) -> PyResult<Py<PyAny>> {
        // Receiver ownership stays in Rust. Python receives only an asyncio.Future
        // that is completed from the runtime thread via call_soon_threadsafe.
        let receiver = {
            let receivers = self.inner.receivers.lock().map_err(lock_error)?;
            receivers.get(&agent_name).cloned()
        }
        .ok_or_else(|| PyKeyError::new_err(format!("unknown agent '{agent_name}'")))?;

        let asyncio = py.import("asyncio")?;
        let event_loop = asyncio.call_method0("get_running_loop")?.unbind();
        let future = event_loop
            .bind(py)
            .call_method0("create_future")?
            .unbind();
        let future_for_python = future.clone_ref(py);

        self.inner.handle.spawn(async move {
            let message = {
                let mut receiver = receiver.lock().await;
                receiver.recv().await
            };

            let _ = Python::attach(|py| match message {
                Some(message) => {
                    let result = Py::new(py, message)?.into_any();
                    schedule_future_result(py, &event_loop, &future, result)
                }
                None => schedule_future_exception(
                    py,
                    &event_loop,
                    &future,
                    "mailbox receiver was closed".to_string(),
                ),
            });
        });

        Ok(future_for_python)
    }

    pub fn get_mailbox_metrics(&self) -> PyResult<Vec<(String, usize, usize, String)>> {
        let senders = self.inner.senders.lock().map_err(lock_error)?;
        let routing_methods = self.inner.routing_methods.lock().map_err(lock_error)?;

        let mut metrics = senders
            .iter()
            .map(|(agent_name, sender)| {
                let buffer_size = sender.max_capacity();
                let queue_depth = buffer_size.saturating_sub(sender.capacity());
                let routing_method = routing_methods
                    .get(agent_name)
                    .cloned()
                    .unwrap_or_else(|| "Direct".to_string());

                (agent_name.clone(), queue_depth, buffer_size, routing_method)
            })
            .collect::<Vec<_>>();

        metrics.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(metrics)
    }
}

impl NativeIPCBus {
    fn resolve_delivery(
        &self,
        receiver_or_capability: &str,
    ) -> PyResult<(String, mpsc::Sender<AgentMessage>, String)> {
        {
            let senders = self.inner.senders.lock().map_err(lock_error)?;
            if let Some(sender) = senders.get(receiver_or_capability) {
                return Ok((
                    receiver_or_capability.to_string(),
                    sender.clone(),
                    "Direct".to_string(),
                ));
            }
        }

        let candidates = self
            .inner
            .kernel
            .find_agents_by_capability(receiver_or_capability)?;

        let senders = self.inner.senders.lock().map_err(lock_error)?;
        for agent_name in candidates {
            if let Some(sender) = senders.get(&agent_name) {
                return Ok((agent_name, sender.clone(), "Semantic Fallback".to_string()));
            }
        }

        Err(PyKeyError::new_err(format!(
            "unknown receiver or capability '{receiver_or_capability}'"
        )))
    }
}

#[derive(Default)]
struct KernelInner {
    capability_agents: Mutex<HashMap<String, BTreeSet<String>>>,
    registered_agents: Mutex<BTreeSet<String>>,
    is_shutting_down: Mutex<bool>,
}

impl KernelInner {
    fn register_agent(&self, agent_name: String) -> PyResult<()> {
        if agent_name.is_empty() {
            return Err(PyValueError::new_err("agent_name must not be empty"));
        }

        self.registered_agents
            .lock()
            .map_err(lock_error)?
            .insert(agent_name);

        Ok(())
    }

    fn register_agent_capability(&self, agent_name: String, capability: String) -> PyResult<()> {
        if agent_name.is_empty() {
            return Err(PyValueError::new_err("agent_name must not be empty"));
        }
        if capability.is_empty() {
            return Err(PyValueError::new_err("capability must not be empty"));
        }

        let mut capability_agents = self.capability_agents.lock().map_err(lock_error)?;
        capability_agents
            .entry(capability)
            .or_default()
            .insert(agent_name.clone());
        self.register_agent(agent_name)?;

        Ok(())
    }

    fn unregister_agent(&self, agent_name: &str) -> PyResult<()> {
        self.registered_agents
            .lock()
            .map_err(lock_error)?
            .remove(agent_name);

        let mut capability_agents = self.capability_agents.lock().map_err(lock_error)?;
        capability_agents.retain(|_, agents| {
            agents.remove(agent_name);
            !agents.is_empty()
        });

        Ok(())
    }

    fn find_agents_by_capability(&self, capability: &str) -> PyResult<Vec<String>> {
        let capability_agents = self.capability_agents.lock().map_err(lock_error)?;
        Ok(capability_agents
            .get(capability)
            .map(|agents| agents.iter().cloned().collect())
            .unwrap_or_default())
    }

    fn total_registered_agents(&self) -> PyResult<usize> {
        Ok(self.registered_agents.lock().map_err(lock_error)?.len())
    }
}

struct BusInner {
    kernel: Arc<KernelInner>,
    handle: Handle,
    senders: Mutex<HashMap<String, mpsc::Sender<AgentMessage>>>,
    receivers: Mutex<HashMap<String, MailboxReceiver>>,
    routing_methods: Mutex<HashMap<String, String>>,
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl BusInner {
    fn start(kernel: Arc<KernelInner>) -> PyResult<Self> {
        let (handle_tx, handle_rx) = std_mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let worker = thread::Builder::new()
            .name("agent-os-ipc-runtime".to_string())
            .spawn(move || {
                let runtime = match Builder::new_multi_thread()
                    .enable_all()
                    .thread_name("agent-os-ipc-worker")
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(err) => {
                        let _ = handle_tx.send(Err(format!(
                            "failed to create Tokio runtime: {err}"
                        )));
                        return;
                    }
                };

                let handle = runtime.handle().clone();
                if handle_tx.send(Ok(handle)).is_err() {
                    return;
                }

                runtime.block_on(async {
                    let _ = shutdown_rx.await;
                });
            })
            .map_err(|err| PyRuntimeError::new_err(format!("failed to spawn IPC worker: {err}")))?;

        let handle = handle_rx
            .recv()
            .map_err(|err| PyRuntimeError::new_err(format!("IPC worker did not start: {err}")))?
            .map_err(PyRuntimeError::new_err)?;

        Ok(Self {
            kernel,
            handle,
            senders: Mutex::new(HashMap::new()),
            receivers: Mutex::new(HashMap::new()),
            routing_methods: Mutex::new(HashMap::new()),
            shutdown: Mutex::new(Some(shutdown_tx)),
            worker: Mutex::new(Some(worker)),
        })
    }
}

impl Drop for BusInner {
    fn drop(&mut self) {
        if let Ok(mut shutdown) = self.shutdown.lock() {
            if let Some(shutdown) = shutdown.take() {
                let _ = shutdown.send(());
            }
        }

        if let Ok(mut worker) = self.worker.lock() {
            if let Some(worker) = worker.take() {
                let _ = worker.join();
            }
        }
    }
}

fn validate_json_payload(payload: &str) -> PyResult<()> {
    serde_json::from_str::<serde_json::Value>(payload)
        .map(|_| ())
        .map_err(|err| PyValueError::new_err(format!("payload must be valid JSON: {err}")))
}

fn lock_error<T>(err: std::sync::PoisonError<T>) -> PyErr {
    PyRuntimeError::new_err(format!("IPC bus lock poisoned: {err}"))
}

fn schedule_future_result(
    py: Python<'_>,
    event_loop: &Py<PyAny>,
    future: &Py<PyAny>,
    result: Py<PyAny>,
) -> PyResult<()> {
    if future.bind(py).call_method0("cancelled")?.is_truthy()? {
        return Ok(());
    }

    let callback = future.bind(py).getattr("set_result")?;
    event_loop
        .bind(py)
        .call_method1("call_soon_threadsafe", (callback, result))?;

    Ok(())
}

fn schedule_future_exception(
    py: Python<'_>,
    event_loop: &Py<PyAny>,
    future: &Py<PyAny>,
    message: String,
) -> PyResult<()> {
    if future.bind(py).call_method0("cancelled")?.is_truthy()? {
        return Ok(());
    }

    let callback = future.bind(py).getattr("set_exception")?;
    let exception = py
        .get_type::<PyRuntimeError>()
        .call1((message,))?
        .unbind();
    event_loop
        .bind(py)
        .call_method1("call_soon_threadsafe", (callback, exception))?;

    Ok(())
}
