//! Dreamer subsystem — memory consolidation via two agentic specialists
//! (deduction then induction) that explore the observation space with tools and
//! WRITE new higher-level observations. Port of Python `src/dreamer/`.
//!
//! Surprisal-based pre-sampling (`src/dreamer/surprisal.py` + `trees/`) is
//! intentionally NOT ported: it is disabled by default (`DREAM.SURPRISAL.ENABLED
//! = False`), the default tree (`kdtree`) depends on sklearn, and even the
//! self-contained `rptree` uses unseeded `np.random` so there is no parity
//! target. A dream simply runs without surprisal hints, which is the
//! default-config behavior.

pub mod specialists;
pub mod tools;
