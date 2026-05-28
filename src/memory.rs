use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_ACTIVE_TOKENS: usize = 8_000;
const PAGE_OUT_TARGET_NUMERATOR: usize = 7;
const PAGE_OUT_TARGET_DENOMINATOR: usize = 10;

#[pyclass]
#[derive(Clone, Debug)]
pub struct MemoryPage {
    #[pyo3(get)]
    pub page_id: String,
    #[pyo3(get)]
    pub timestamp: u64,
    #[pyo3(get)]
    pub token_count: usize,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub is_paged_out: bool,
}

#[pymethods]
impl MemoryPage {
    #[new]
    pub fn new(
        page_id: String,
        timestamp: u64,
        token_count: usize,
        content: String,
        is_paged_out: bool,
    ) -> PyResult<Self> {
        validate_json_content(&content)?;

        Ok(Self {
            page_id,
            timestamp,
            token_count,
            content,
            is_paged_out,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryPage(page_id={:?}, timestamp={}, token_count={}, is_paged_out={})",
            self.page_id, self.timestamp, self.token_count, self.is_paged_out
        )
    }
}

#[derive(Clone, Debug)]
struct AgentPageTable {
    agent_name: String,
    max_active_tokens: usize,
    current_active_tokens: usize,
    pages: Vec<MemoryPage>,
    pending_evictions: Vec<MemoryPage>,
}

impl AgentPageTable {
    fn new(agent_name: String, max_active_tokens: usize) -> Self {
        Self {
            agent_name,
            max_active_tokens,
            current_active_tokens: 0,
            pages: Vec::new(),
            pending_evictions: Vec::new(),
        }
    }

    fn page_out_target_tokens(&self) -> usize {
        self.max_active_tokens * PAGE_OUT_TARGET_NUMERATOR / PAGE_OUT_TARGET_DENOMINATOR
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ContextMemoryManager {
    page_tables: Arc<Mutex<HashMap<String, AgentPageTable>>>,
    default_max_active_tokens: usize,
}

#[pymethods]
impl ContextMemoryManager {
    #[new]
    #[pyo3(signature = (max_active_tokens=DEFAULT_MAX_ACTIVE_TOKENS))]
    pub fn new(max_active_tokens: usize) -> PyResult<Self> {
        if max_active_tokens == 0 {
            return Err(PyValueError::new_err(
                "max_active_tokens must be greater than zero",
            ));
        }

        Ok(Self {
            page_tables: Arc::new(Mutex::new(HashMap::new())),
            default_max_active_tokens: max_active_tokens,
        })
    }

    pub fn register_agent(&self, agent_name: String, max_active_tokens: usize) -> PyResult<()> {
        if agent_name.is_empty() {
            return Err(PyValueError::new_err("agent_name must not be empty"));
        }
        if max_active_tokens == 0 {
            return Err(PyValueError::new_err(
                "max_active_tokens must be greater than zero",
            ));
        }

        let mut page_tables = self.page_tables.lock().map_err(lock_error)?;
        page_tables.insert(
            agent_name.clone(),
            AgentPageTable::new(agent_name, max_active_tokens),
        );

        Ok(())
    }

    pub fn unregister_agent(&self, agent_name: String) -> PyResult<bool> {
        let mut page_tables = self.page_tables.lock().map_err(lock_error)?;
        Ok(page_tables.remove(&agent_name).is_some())
    }

    pub fn append_context_frame(
        &self,
        agent_name: String,
        content: String,
        token_estimate: usize,
    ) -> PyResult<bool> {
        if agent_name.is_empty() {
            return Err(PyValueError::new_err("agent_name must not be empty"));
        }
        if token_estimate == 0 {
            return Err(PyValueError::new_err(
                "token_estimate must be greater than zero",
            ));
        }
        validate_json_content(&content)?;

        let mut page_tables = self.page_tables.lock().map_err(lock_error)?;
        let page_table = page_tables
            .entry(agent_name.clone())
            .or_insert_with(|| AgentPageTable::new(agent_name.clone(), self.default_max_active_tokens));

        let page = MemoryPage {
            page_id: build_page_id(&agent_name, page_table.pages.len()),
            timestamp: unix_timestamp()?,
            token_count: token_estimate,
            content,
            is_paged_out: false,
        };

        page_table.current_active_tokens += token_estimate;
        page_table.pages.push(page);

        if page_table.current_active_tokens > page_table.max_active_tokens {
            let evicted_pages = execute_page_out(page_table);
            page_table.pending_evictions.extend(evicted_pages);
            return Ok(true);
        }

        Ok(false)
    }

    pub fn page_in_frame(
        &self,
        agent_name: String,
        page_id: String,
        restored_content: String,
    ) -> PyResult<()> {
        validate_json_content(&restored_content)?;

        let mut page_tables = self.page_tables.lock().map_err(lock_error)?;
        let page_table = page_tables
            .get_mut(&agent_name)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown agent '{agent_name}'")))?;

        let page_index = page_table
            .pages
            .iter()
            .position(|page| page.page_id == page_id)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown page '{page_id}'")))?;

        let was_paged_out = page_table.pages[page_index].is_paged_out;
        let token_count = page_table.pages[page_index].token_count;
        if was_paged_out {
            page_table.current_active_tokens += token_count;
        }

        let page = &mut page_table.pages[page_index];
        page.content = restored_content;
        page.is_paged_out = false;

        if page_table.current_active_tokens > page_table.max_active_tokens {
            let evicted_pages = execute_page_out(page_table);
            page_table.pending_evictions.extend(evicted_pages);
        }

        Ok(())
    }

    pub fn get_active_context(&self, agent_name: String) -> PyResult<Vec<String>> {
        let page_tables = self.page_tables.lock().map_err(lock_error)?;
        let page_table = page_tables
            .get(&agent_name)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown agent '{agent_name}'")))?;

        Ok(page_table
            .pages
            .iter()
            .filter(|page| !page.is_paged_out)
            .map(|page| page.content.clone())
            .collect())
    }

    pub fn get_page_table_summary(&self, py: Python<'_>, agent_name: String) -> PyResult<PyObject> {
        let page_tables = self.page_tables.lock().map_err(lock_error)?;
        let page_table = page_tables
            .get(&agent_name)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown agent '{agent_name}'")))?;

        let active_frames = page_table
            .pages
            .iter()
            .filter(|page| !page.is_paged_out)
            .count();
        let paged_out_frames = page_table.pages.len() - active_frames;

        let summary = PyDict::new_bound(py);
        summary.set_item("agent_name", &page_table.agent_name)?;
        summary.set_item("current_active_tokens", page_table.current_active_tokens)?;
        summary.set_item("max_active_tokens", page_table.max_active_tokens)?;
        summary.set_item("active_frames", active_frames)?;
        summary.set_item("paged_out_frames", paged_out_frames)?;
        summary.set_item("total_frames", page_table.pages.len())?;
        summary.set_item("pending_evictions", page_table.pending_evictions.len())?;

        Ok(summary.into_py(py))
    }

    pub fn list_agents(&self) -> PyResult<Vec<String>> {
        let page_tables = self.page_tables.lock().map_err(lock_error)?;
        let mut agents = page_tables.keys().cloned().collect::<Vec<_>>();
        agents.sort();
        Ok(agents)
    }

    pub fn get_global_active_token_count(&self) -> PyResult<usize> {
        let page_tables = self.page_tables.lock().map_err(lock_error)?;
        Ok(page_tables
            .values()
            .map(|page_table| page_table.current_active_tokens)
            .sum())
    }

    pub fn take_evicted_pages(&self, agent_name: String) -> PyResult<Vec<MemoryPage>> {
        let mut page_tables = self.page_tables.lock().map_err(lock_error)?;
        let page_table = page_tables
            .get_mut(&agent_name)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown agent '{agent_name}'")))?;

        Ok(std::mem::take(&mut page_table.pending_evictions))
    }
}

fn execute_page_out(page_table: &mut AgentPageTable) -> Vec<MemoryPage> {
    let target_tokens = page_table.page_out_target_tokens();
    let mut evicted_pages = Vec::new();

    for page in page_table.pages.iter_mut() {
        if page_table.current_active_tokens <= target_tokens {
            break;
        }

        if page.is_paged_out {
            continue;
        }

        page_table.current_active_tokens =
            page_table.current_active_tokens.saturating_sub(page.token_count);
        page.is_paged_out = true;

        let evicted_page = page.clone();
        page.content = serde_json::json!({
            "page_id": page.page_id,
            "status": "paged_out",
            "token_count": page.token_count,
        })
        .to_string();
        evicted_pages.push(evicted_page);
    }

    evicted_pages
}

fn build_page_id(agent_name: &str, page_index: usize) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();

    format!("{agent_name}-{millis}-{page_index}")
}

fn unix_timestamp() -> PyResult<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|err| PyRuntimeError::new_err(format!("system clock is before UNIX epoch: {err}")))
}

fn validate_json_content(content: &str) -> PyResult<()> {
    serde_json::from_str::<serde_json::Value>(content)
        .map(|_| ())
        .map_err(|err| PyValueError::new_err(format!("content must be valid JSON: {err}")))
}

fn lock_error<T>(err: std::sync::PoisonError<T>) -> PyErr {
    PyRuntimeError::new_err(format!("memory manager lock poisoned: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_page_out_drops_active_tokens_below_target() {
        let mut page_table = AgentPageTable::new("Agent_Test".to_string(), 100);

        for index in 0..6 {
            page_table.pages.push(MemoryPage {
                page_id: format!("page-{index}"),
                timestamp: index,
                token_count: 20,
                content: format!("{{\"index\":{index}}}"),
                is_paged_out: false,
            });
            page_table.current_active_tokens += 20;
        }

        let evicted = execute_page_out(&mut page_table);

        assert_eq!(page_table.current_active_tokens, 60);
        assert_eq!(evicted.len(), 3);
        assert!(page_table.pages[0].is_paged_out);
        assert!(page_table.pages[1].is_paged_out);
        assert!(page_table.pages[2].is_paged_out);
        assert!(!page_table.pages[3].is_paged_out);
    }

    #[test]
    fn page_out_skips_already_evicted_pages() {
        let mut page_table = AgentPageTable::new("Agent_Test".to_string(), 100);

        page_table.pages.push(MemoryPage {
            page_id: "old".to_string(),
            timestamp: 1,
            token_count: 50,
            content: "{\"status\":\"paged_out\"}".to_string(),
            is_paged_out: true,
        });

        for index in 0..4 {
            page_table.pages.push(MemoryPage {
                page_id: format!("active-{index}"),
                timestamp: index + 2,
                token_count: 30,
                content: format!("{{\"index\":{index}}}"),
                is_paged_out: false,
            });
            page_table.current_active_tokens += 30;
        }

        let evicted = execute_page_out(&mut page_table);

        assert_eq!(page_table.current_active_tokens, 60);
        assert_eq!(evicted.len(), 2);
        assert!(page_table.pages[0].is_paged_out);
        assert!(page_table.pages[1].is_paged_out);
        assert!(page_table.pages[2].is_paged_out);
        assert!(!page_table.pages[3].is_paged_out);
    }
}
