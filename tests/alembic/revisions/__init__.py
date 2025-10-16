"""Register revision-specific hooks for migration verification."""

from . import (
    test_05486ce795d5_make_session_name_required_on_messages,
    test_08894082221a_replace_collection_name_with_observer_,
    test_20f89a421aff_rename_metamessage_type_to_label,
    test_66e63cf2cf77_add_indexes_to_documents_table,
    test_76ffba56fe8c_add_error_field_to_queueitem,
    test_88b0fb10906f_add_webhooks_table,
    test_556a16564f50_add_user_id_and_app_id_to_tables,
    test_564ba40505c5_add_session_name_column_to_documents,
    test_917195d9b5e9_add_messageembedding_table,
    test_a1b2c3d4e5f6_initial_schema,
    test_b765d82110bd_change_metamessages_to_user_level_with_,
    test_c3828084f472_add_indexes_for_messages_and_,
    test_d429de0e5338_adopt_peer_paradigm,
)

__all__ = [
    "test_05486ce795d5_make_session_name_required_on_messages",
    "test_08894082221a_replace_collection_name_with_observer_",
    "test_20f89a421aff_rename_metamessage_type_to_label",
    "test_66e63cf2cf77_add_indexes_to_documents_table",
    "test_76ffba56fe8c_add_error_field_to_queueitem",
    "test_88b0fb10906f_add_webhooks_table",
    "test_556a16564f50_add_user_id_and_app_id_to_tables",
    "test_564ba40505c5_add_session_name_column_to_documents",
    "test_917195d9b5e9_add_messageembedding_table",
    "test_a1b2c3d4e5f6_initial_schema",
    "test_b765d82110bd_change_metamessages_to_user_level_with_",
    "test_c3828084f472_add_indexes_for_messages_and_",
    "test_d429de0e5338_adopt_peer_paradigm",
]
