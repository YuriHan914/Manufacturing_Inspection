from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import toml

ROOT_DIR = Path(__file__).resolve().parents[1]
SECRETS_PATH = ROOT_DIR / ".streamlit" / "secrets.toml"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.utils import (
    DEFAULT_LLM_MAX_NEW_TOKENS,
    DEFAULT_LLM_MODEL_DIR,
    DEFAULT_LLM_TEMPERATURE,
    _append_app_log,
    _format_display_path,
    _get_llm_runtime_settings,
    _get_pending_llm_runtime_settings,
    _looks_like_project_path,
    _to_project_relative_path,
    configure_page,
    is_model_downloaded,
    load_dashboard_data,
    render_page_header,
)


def _read_toml_file(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = default or {}
    if not path.exists():
        return fallback
    try:
        return toml.loads(path.read_text(encoding="utf-8"))
    except (toml.TomlDecodeError, OSError):
        return fallback


def _get_supabase_secret_settings() -> dict[str, str]:
    secrets_config = _read_toml_file(SECRETS_PATH, {})
    connection_config = secrets_config.get("connections", {}).get("supabase", {})
    return {
        "url": str(connection_config.get("SUPABASE_URL", "") or "").strip(),
        "key": str(connection_config.get("SUPABASE_KEY", "") or "").strip(),
    }


def _write_supabase_secret_settings(url: str, key: str) -> None:
    secrets_config = _read_toml_file(SECRETS_PATH, {})
    connections = secrets_config.setdefault("connections", {})
    supabase = connections.setdefault("supabase", {})
    supabase["SUPABASE_URL"] = url.strip()
    supabase["SUPABASE_KEY"] = key.strip()
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_PATH.write_text(toml.dumps(secrets_config), encoding="utf-8")


def _mask_secret_value(value: str, *, keep_start: int = 6, keep_end: int = 4) -> str:
    if not value:
        return "-"
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return f"{value[:keep_start]}{'*' * 8}{value[-keep_end:]}"


def render_setting_page(config) -> None:
    render_page_header("Setting")
    top_left, _ = st.columns([1.1, 1.9])
    current_supabase_settings = _get_supabase_secret_settings()
    st.session_state.setdefault("db_setting_panel_open", False)
    st.session_state.setdefault("db_setting_status", "idle")
    st.session_state.setdefault("db_setting_notice", "")
    st.session_state.setdefault("supabase_url_pending", current_supabase_settings["url"])
    st.session_state.setdefault("supabase_key_pending", current_supabase_settings["key"])

    with top_left:
        if st.button("DB Settings", key="db_setting_toggle_button", width="stretch"):
            st.session_state["db_setting_panel_open"] = not bool(st.session_state["db_setting_panel_open"])
            st.session_state["supabase_url_pending"] = current_supabase_settings["url"]
            st.session_state["supabase_key_pending"] = current_supabase_settings["key"]
            st.session_state["db_setting_status"] = "idle"
            st.session_state["db_setting_notice"] = ""
            st.rerun()

    if st.session_state["db_setting_panel_open"]:
        with st.container(border=True):
            st.subheader("Supabase DB")
            st.caption(
                "Edit the `connections.supabase` values in `.streamlit/secrets.toml` from the UI. "
                "Restarting the app after saving is the most reliable option."
            )
            st.text_input(
                "Supabase URL",
                key="supabase_url_pending",
                placeholder="http://127.0.0.1:54321",
            )
            st.text_input(
                "Supabase KEY",
                key="supabase_key_pending",
                type="password",
                placeholder="service_role or anon key",
            )
            button_col, close_col = st.columns(2, gap="large")
            with button_col:
                save_db_clicked = st.button(
                    "Save DB Settings",
                    key="db_setting_apply",
                    type="primary",
                    width="stretch",
                )
            with close_col:
                close_db_clicked = st.button(
                    "Close",
                    key="db_setting_close",
                    width="stretch",
                )

            if save_db_clicked:
                try:
                    supabase_url = str(st.session_state.get("supabase_url_pending", "")).strip()
                    supabase_key = str(st.session_state.get("supabase_key_pending", "")).strip()
                    if not supabase_url:
                        raise RuntimeError("Please enter the Supabase URL.")
                    if not supabase_key:
                        raise RuntimeError("Please enter the Supabase key.")

                    _write_supabase_secret_settings(supabase_url, supabase_key)
                    load_dashboard_data.clear()
                    st.cache_data.clear()
                    st.session_state["db_setting_status"] = "done"
                    st.session_state["db_setting_notice"] = "The Supabase DB settings have been saved."
                    _append_app_log(
                        log_type="done",
                        source="Setting",
                        content="Supabase DB settings were saved.",
                        request=f"url={supabase_url}",
                        response=f"key={_mask_secret_value(supabase_key)}",
                    )
                    st.rerun()
                except Exception as exc:
                    st.session_state["db_setting_status"] = "error"
                    st.session_state["db_setting_notice"] = "Failed to save the Supabase DB settings."
                    _append_app_log(
                        log_type="error",
                        source="Setting",
                        content="Supabase DB settings could not be saved.",
                        request=f"url={str(st.session_state.get('supabase_url_pending', '')).strip()}",
                        response=str(exc),
                    )
                    st.rerun()

            if close_db_clicked:
                st.session_state["db_setting_panel_open"] = False
                st.session_state["db_setting_status"] = "idle"
                st.session_state["db_setting_notice"] = ""
                st.rerun()

            db_status = st.session_state["db_setting_status"]
            if db_status == "done":
                st.success(st.session_state["db_setting_notice"])
            elif db_status == "error":
                st.error(st.session_state["db_setting_notice"])

            st.caption(f"Secrets file: {_to_project_relative_path(SECRETS_PATH)}")
            st.caption(f"Saved URL: {current_supabase_settings['url'] or '-'}")
            st.caption(f"Saved KEY: {_mask_secret_value(current_supabase_settings['key'])}")

    applied_llm_settings = _get_llm_runtime_settings()
    pending_llm_settings = _get_pending_llm_runtime_settings()
    available_model_dirs = applied_llm_settings["available_model_dirs"]
    available_model_values = [str(path) for path in available_model_dirs]
    st.session_state.setdefault("llm_setting_status", "idle")
    st.session_state.setdefault("llm_setting_notice", "")

    with st.container(border=True):
        st.subheader("LLM Runtime")
        st.caption("These settings control the model used for sidebar LLM responses and Summary analysis comments.")

        if available_model_values:
            current_model_value = str(pending_llm_settings["model_dir"])
            if current_model_value not in available_model_values:
                current_model_value = available_model_values[0]

            model_col, temp_col, token_col = st.columns(3, gap="large")
            with model_col:
                selected_model = st.selectbox(
                    "LLM model",
                    options=available_model_values,
                    index=available_model_values.index(current_model_value),
                    format_func=lambda value: Path(value).name,
                    key="llm_model_dir_pending",
                )
            with temp_col:
                st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.5,
                    value=float(pending_llm_settings["temperature"]),
                    step=0.1,
                    key="llm_temperature_pending",
                )
            with token_col:
                st.number_input(
                    "Max tokens",
                    min_value=64,
                    max_value=4096,
                    value=int(pending_llm_settings["max_new_tokens"]),
                    step=64,
                    key="llm_max_new_tokens_pending",
                )

            apply_clicked = st.button("Set", key="llm_runtime_apply", type="primary", width="stretch")
            if apply_clicked:
                try:
                    pending_model_dir = Path(str(st.session_state["llm_model_dir_pending"]))
                    if not available_model_values:
                        raise RuntimeError("No local LLM models are available.")
                    if str(pending_model_dir) not in available_model_values:
                        raise RuntimeError("The selected LLM model could not be found.")
                    if not is_model_downloaded(pending_model_dir):
                        raise RuntimeError("The selected LLM model files are not fully prepared yet.")

                    st.session_state["llm_model_dir"] = str(pending_model_dir)
                    st.session_state["llm_temperature"] = max(
                        0.0,
                        float(st.session_state["llm_temperature_pending"]),
                    )
                    st.session_state["llm_max_new_tokens"] = max(
                        1,
                        int(st.session_state["llm_max_new_tokens_pending"]),
                    )
                    st.session_state["llm_setting_status"] = "done"
                    st.session_state["llm_setting_notice"] = "The LLM runtime settings have been applied."
                    _append_app_log(
                        log_type="done",
                        source="Setting",
                        content="LLM runtime settings were applied successfully.",
                        request=(
                            f"model={_to_project_relative_path(pending_model_dir)}, "
                            f"temperature={float(st.session_state['llm_temperature']):.1f}, "
                            f"max_tokens={int(st.session_state['llm_max_new_tokens'])}"
                        ),
                    )
                except Exception as exc:
                    st.session_state["llm_setting_status"] = "error"
                    st.session_state["llm_setting_notice"] = "Failed to apply the LLM runtime settings."
                    _append_app_log(
                        log_type="error",
                        source="Setting",
                        content="LLM runtime settings could not be applied.",
                        request=(
                            f"model={_format_display_path(st.session_state.get('llm_model_dir_pending'))}, "
                            f"temperature={float(st.session_state.get('llm_temperature_pending', DEFAULT_LLM_TEMPERATURE)):.1f}, "
                            f"max_tokens={int(st.session_state.get('llm_max_new_tokens_pending', DEFAULT_LLM_MAX_NEW_TOKENS))}"
                        ),
                        response=str(exc),
                    )
                st.rerun()

            status = st.session_state["llm_setting_status"]
            if status == "done":
                st.success(st.session_state["llm_setting_notice"])
            elif status == "error":
                st.error(st.session_state["llm_setting_notice"])

            st.caption(
                "Applied setting: "
                f"{Path(str(applied_llm_settings['model_dir'])).name} | "
                f"Temp {float(applied_llm_settings['temperature']):.1f} | "
                f"Max tokens {int(applied_llm_settings['max_new_tokens'])}"
            )
            st.caption(f"Applied model path: {_to_project_relative_path(applied_llm_settings['model_dir'])}")
            st.caption(f"Pending model path: {_to_project_relative_path(selected_model)}")
        else:
            st.warning(f"No available local LLM models were found: {_to_project_relative_path(DEFAULT_LLM_MODEL_DIR.parent)}")

    with st.expander("Current configuration", expanded=False):
        config_frame = pd.DataFrame(
            [
                {
                    "Setting": key,
                    "Value": _format_display_path(value) if _looks_like_project_path(value) else str(value),
                }
                for key, value in config.items()
                if key != "classes_to_train"
            ]
        )
        st.dataframe(config_frame, width="stretch", hide_index=True)
        if isinstance(config.get("classes_to_train"), list):
            st.write("Classes To Train")
            st.write(", ".join(str(item) for item in config["classes_to_train"]))


configure_page("Setting")
config, _runs, _image_records, _log_entries = load_dashboard_data()
render_setting_page(config)
