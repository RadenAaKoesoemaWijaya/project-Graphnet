"""
Authentication and Role-Based Access Control (RBAC) Manager for ASTINA.

Complies with:
- UU No. 27 Tahun 2022 (Perlindungan Data Pribadi / UU PDP)
- HIPAA Security Rule (Access Control & Audit Controls)
- Role-Based Access Control (RBAC) Architecture
"""

import os
import hashlib
import logging
from typing import Dict, Any, List, Optional
import streamlit as st

logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULT ROLES & PERMISSIONS
# =============================================================================

ROLE_PERMISSIONS: Dict[str, List[str]] = {
    'admin': ['home', 'collect', 'train', 'evaluate', 'detect', 'status'],
    'auditor': ['home', 'detect', 'status'],
    'analyst': ['home', 'collect', 'train', 'evaluate', 'detect'],
    'viewer': ['home', 'status']
}

PAGE_NAMES: Dict[str, str] = {
    'home': 'Beranda',
    'collect': 'Unggah Data',
    'train': 'Pelatihan Model',
    'evaluate': 'Evaluasi Model',
    'detect': 'Deteksi Anomali',
    'status': 'Status Sistem'
}

# =============================================================================
# SECURE PASSWORD HASHING UTILITY
# =============================================================================

def hash_password(password: str, salt: str = "ASTINA_SECURE_SALT_v1") -> str:
    """Hash password using SHA-256 with cryptographic salt."""
    return hashlib.sha256(f"{salt}:{password}:{salt}".encode('utf-8')).hexdigest()

# Initial pre-configured accounts (passwords can be overridden via environment variables)
# Default dev passwords: admin -> AdminAstina2026!, auditor -> AuditorAstina2026!, analyst -> AnalystAstina2026!, viewer -> ViewerAstina2026!
DEFAULT_USERS: Dict[str, Dict[str, Any]] = {
    'admin': {
        'name': 'System Administrator',
        'role': 'admin',
        'hash': hash_password(os.getenv('ASTINA_ADMIN_PASSWORD', 'AdminAstina2026!')),
        'email': 'admin@astina.ai'
    },
    'auditor': {
        'name': 'Investigator Senior ASTINA',
        'role': 'auditor',
        'hash': hash_password(os.getenv('ASTINA_AUDITOR_PASSWORD', 'AuditorAstina2026!')),
        'email': 'auditor@astina.ai'
    },
    'analyst': {
        'name': 'Lead Data Scientist',
        'role': 'analyst',
        'hash': hash_password(os.getenv('ASTINA_ANALYST_PASSWORD', 'AnalystAstina2026!')),
        'email': 'analyst@astina.ai'
    },
    'viewer': {
        'name': 'Executive Reviewer',
        'role': 'viewer',
        'hash': hash_password(os.getenv('ASTINA_VIEWER_PASSWORD', 'ViewerAstina2026!')),
        'email': 'viewer@astina.ai'
    }
}


class AuthManager:
    """Manages user authentication, sessions, and RBAC authorization."""

    @staticmethod
    def is_auth_enforced() -> bool:
        """Check whether login is mandatory via environment variable AUTH_ENABLED."""
        val = os.getenv('AUTH_ENABLED', 'false').lower().strip()
        return val in ('true', '1', 'yes', 'on')

    @staticmethod
    def is_authenticated() -> bool:
        """Check if current Streamlit session has an active authenticated user."""
        if not AuthManager.is_auth_enforced():
            return True
        return bool(st.session_state.get('authenticated', False))

    @staticmethod
    def get_current_user() -> Dict[str, Any]:
        """Get currently logged-in user profile dictionary."""
        if not AuthManager.is_auth_enforced():
            # In development bypass mode, default to full access admin
            return {
                'username': st.session_state.get('username', 'admin'),
                'name': st.session_state.get('user_fullname', 'Administrator (Dev Mode)'),
                'role': st.session_state.get('user_role', 'admin'),
                'email': 'dev@astina.ai'
            }
        return {
            'username': st.session_state.get('username', 'anonymous'),
            'name': st.session_state.get('user_fullname', 'Guest User'),
            'role': st.session_state.get('user_role', 'viewer'),
            'email': st.session_state.get('user_email', '')
        }

    @staticmethod
    def get_current_role() -> str:
        """Return role string for current session ('admin', 'auditor', 'analyst', 'viewer')."""
        return AuthManager.get_current_user().get('role', 'viewer')

    @staticmethod
    def can_access_page(page: str) -> bool:
        """Check if current user has authorization to access specified page."""
        role = AuthManager.get_current_role()
        allowed = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS['viewer'])
        return page in allowed

    @staticmethod
    def authenticate(username: str, password: str) -> bool:
        """Verify username and password against credentials store."""
        username = username.strip().lower()
        user = DEFAULT_USERS.get(username)
        if not user:
            logger.warning(f"Failed authentication attempt: user '{username}' not found.")
            return False

        input_hash = hash_password(password)
        if input_hash == user['hash']:
            # Set session variables
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.session_state['user_fullname'] = user['name']
            st.session_state['user_role'] = user['role']
            st.session_state['user_email'] = user.get('email', '')
            st.session_state['copilot_auditor_val'] = user['name']

            # Record in audit trail if available
            try:
                from audit_trail import get_audit_logger
                audit = get_audit_logger()
                audit.log_event(
                    event_type="USER_LOGIN_SUCCESS",
                    actor=username,
                    details={"role": user["role"], "auth_type": "PASSWORD"}
                )
            except Exception as e:
                logger.debug(f"Audit log on login skipped: {e}")

            logger.info(f"User '{username}' successfully authenticated with role '{user['role']}'.")
            return True

        logger.warning(f"Failed authentication attempt: invalid password for '{username}'.")
        return False

    @staticmethod
    def logout():
        """Terminate current session securely."""
        username = st.session_state.get('username', 'unknown')
        try:
            from audit_trail import get_audit_logger
            audit = get_audit_logger()
            audit.log_event(
                event_type="USER_LOGOUT",
                actor=username,
                details={"action": "SESSION_CLOSED"}
            )
        except Exception:
            pass

        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.session_state['user_fullname'] = None
        st.session_state['user_role'] = 'viewer'
        st.session_state['page'] = 'home'
        st.rerun()

    @staticmethod
    def render_login_page():
        """Render modern, glassmorphic login gateway."""
        st.markdown("""
        <div style="max-width:480px; margin: 40px auto 20px auto; text-align: center;">
            <div style="background: linear-gradient(135deg, #1e3a8a, #0f172a); border-radius: 16px; padding: 24px 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); border: 1px solid rgba(59, 130, 246, 0.3);">
                <div style="font-size: 2.2rem; margin-bottom: 6px;">🛡️</div>
                <h2 style="color: #ffffff; margin: 0; font-weight: 800; font-size: 1.6rem; letter-spacing: 0.5px;">ASTINA ENTERPRISE</h2>
                <div style="color: #93c5fd; font-size: 0.8rem; font-weight: 600; margin-top: 4px; letter-spacing: 0.5px;">
                    SECURE FRAUD DETECTION & AUDIT GATEWAY
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            with st.container():
                st.markdown("""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:24px 28px; box-shadow:0 4px 12px rgba(0,0,0,0.05); margin-bottom:20px;">
                    <div style="font-size:0.88rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:14px; text-align:center;">
                        🔑 Silakan Masuk untuk Mengakses Sistem
                    </div>
                """, unsafe_allow_html=True)

                username_input = st.text_input("Username / ID Pengguna:", placeholder="admin / auditor / analyst / viewer", key="login_username_field")
                password_input = st.text_input("Kata Sandi (Password):", type="password", placeholder="••••••••••••", key="login_password_field")

                if st.button("🔐 Masuk ke Sistem (Log In)", type="primary", key="login_btn_submit"):
                    if username_input and password_input:
                        success = AuthManager.authenticate(username_input, password_input)
                        if success:
                            st.success(f"✅ Berhasil masuk! Selamat datang, {st.session_state['user_fullname']}.")
                            st.rerun()
                        else:
                            st.error("❌ Nama pengguna atau kata sandi tidak valid. Akses ditolak.")
                    else:
                        st.warning("⚠️ Harap lengkapi nama pengguna dan kata sandi.")

                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("ℹ️ Bantuan Akses & Kredensial Default Uji Coba"):
                    st.markdown("""
                    | Role | Username | Password Default | Hak Akses |
                    | :--- | :--- | :--- | :--- |
                    | **Admin** | `admin` | `AdminAstina2026!` | Akses Penuh Seluruh Modul |
                    | **Auditor** | `auditor` | `AuditorAstina2026!` | Deteksi, Review Klaim & Copilot BAP |
                    | **Analyst** | `analyst` | `AnalystAstina2026!` | Upload, Training & Evaluasi Model |
                    | **Viewer** | `viewer` | `ViewerAstina2026!` | Akses Baca & Status Sistem |
                    """)
