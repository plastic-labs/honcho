#[cfg(feature = "blocking")]
static_assertions::assert_impl_all!(
    honcho_ai::blocking::Honcho: Send, Sync, Clone
);

#[cfg(feature = "blocking")]
static_assertions::assert_impl_all!(
    honcho_ai::blocking::Peer: Send, Sync, Clone
);

#[cfg(feature = "blocking")]
static_assertions::assert_impl_all!(
    honcho_ai::blocking::Session: Send, Sync, Clone
);

#[cfg(feature = "blocking")]
static_assertions::assert_impl_all!(
    honcho_ai::blocking::Conclusion: Send, Sync, Clone
);

#[cfg(feature = "blocking")]
static_assertions::assert_impl_all!(
    honcho_ai::blocking::ConclusionScope: Send, Sync, Clone
);
