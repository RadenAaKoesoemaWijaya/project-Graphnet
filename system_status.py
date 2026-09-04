"""
System status dashboard for ASTINA application.
Displays real-time system health, performance metrics, and audit information.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

def show_system_status_page():
    """Display comprehensive system status dashboard"""
    st.title("🖥️ Status Sistem")
    st.markdown("Monitoring real-time kesehatan dan performa sistem ASTINA")
    
    # Initialize modules
    try:
        from enhanced_metrics import get_metrics_collector, get_system_status
        from audit_trail import get_audit_trail
        from error_handler import ErrorContext
        from ui.utils import get_gpu_status
        
        metrics_collector = get_metrics_collector()
        audit_trail = get_audit_trail()
        
        # Refresh button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            st.caption(f"Terakhir diupdate: {datetime.now().strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # GPU Status Section
        st.subheader("⚙️ Konfigurasi Perangkat")
        
        gpu_info = get_gpu_status()
        
        gpu_cols = st.columns(2)
        with gpu_cols[0]:
            st.metric("PyTorch Version", gpu_info['torch_version'])
        with gpu_cols[1]:
            st.metric("Device", gpu_info['device_name'])
        
        if gpu_info['cuda_available']:
            gpu_detail_cols = st.columns(3)
            with gpu_detail_cols[0]:
                st.metric("GPU Status", "✅ Active")
            with gpu_detail_cols[1]:
                st.metric("Device Count", gpu_info['device_count'])
            with gpu_detail_cols[2]:
                if gpu_info['total_memory'] > 0:
                    st.metric("Total Memory", f"{gpu_info['total_memory']:.1f} GB")
            
            if gpu_info['compute_capability']:
                st.info(f"📊 Compute Capability: {gpu_info['compute_capability']}")
        else:
            st.warning("💻 Menggunakan CPU Mode - Untuk performa lebih baik, install PyTorch dengan GPU support (CUDA/ROCm)")
            with st.expander("📖 Lihat Panduan Instalasi GPU (NVIDIA CUDA)"):
                st.markdown("""
                Sistem mendeteksi bahwa PyTorch saat ini berjalan dalam **Mode CPU**. Untuk mengaktifkan akselerasi GPU:
                
                **Langkah Cepat (PowerShell):**
                ```powershell
                # 1. Aktifkan venv
                .\\.venv\\Scripts\\Activate.ps1
                # 2. Install PyTorch dengan dukungan CUDA 12.4
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
                # 3. Verifikasi ketersediaan CUDA
                python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
                ```
                
                *Panduan lengkap arsitektur dan optimasi VRAM dapat dilihat pada file `README.md` (bagian **Panduan Setup Akselerasi GPU**).*
                """)
        
        st.markdown("---")
        
        # System Health Section
        st.subheader("🏥 Kesehatan Sistem")
        
        health_status = metrics_collector.health.run_all_checks()
        
        if health_status['overall_status'] == 'healthy':
            st.success("✅ Sistem dalam kondisi sehat")
        else:
            st.error("❌ Sistem mengalami masalah")
        
        # Display individual health checks
        health_cols = st.columns(len(health_status['checks']))
        for i, (check_name, check_result) in enumerate(health_status['checks'].items()):
            with health_cols[i]:
                status_icon = "✅" if check_result['status'] == 'healthy' else "❌"
                st.metric(
                    f"{status_icon} {check_name.replace('_', ' ').title()}",
                    check_result['status'].upper(),
                    f"{check_result['duration']:.3f}s"
                )
        
        st.markdown("---")
        
        # Performance Metrics Section
        st.subheader("📊 Metrik Performa")
        
        all_metrics = metrics_collector.performance.get_all_metrics()
        
        # Display counters
        if all_metrics['counters']:
            st.markdown("**Counters:**")
            counter_cols = st.columns(4)
            for i, (counter_name, counter_value) in enumerate(all_metrics['counters'].items()):
                with counter_cols[i % 4]:
                    st.metric(
                        counter_name.replace('_', ' ').title(),
                        f"{counter_value:,}"
                    )
        
        # Display gauges
        if all_metrics['gauges']:
            st.markdown("**Current Values:**")
            gauge_cols = st.columns(4)
            for i, (gauge_name, gauge_value) in enumerate(all_metrics['gauges'].items()):
                with gauge_cols[i % 4]:
                    display_value = f"{gauge_value:.2f}" if isinstance(gauge_value, float) else f"{gauge_value}"
                    st.metric(
                        gauge_name.replace('_', ' ').title(),
                        display_value
                    )
        
        # Display timing statistics
        if all_metrics['timing_operations']:
            st.markdown("**Timing Statistics:**")
            timing_ops = all_metrics['timing_operations']
            
            for op in timing_ops:
                stats = metrics_collector.performance.get_timing_stats(op)
                if stats:
                    with st.expander(f"📈 {op.replace('_', ' ').title()}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Count", f"{stats['count']:,}")
                        with col2:
                            st.metric("Mean", f"{stats['mean']:.2f}s")
                        with col3:
                            st.metric("P95", f"{stats['p95']:.2f}s")
                        with col4:
                            st.metric("P99", f"{stats['p99']:.2f}s")
        
        st.markdown("---")
        
        # Recent Alerts Section
        st.subheader("🚨 Alert Terbaru")
        
        recent_alerts = metrics_collector.alerts.get_recent_alerts(10)
        
        if recent_alerts:
            alert_df = pd.DataFrame(recent_alerts)
            alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
            
            # Color code by severity
            def color_severity(severity):
                if severity == 'critical':
                    return '🔴'
                elif severity == 'error':
                    return '🟠'
                elif severity == 'warning':
                    return '🟡'
                else:
                    return '🟢'
            
            alert_df['severity_icon'] = alert_df['severity'].apply(color_severity)
            
            display_cols = ['severity_icon', 'metric_name', 'current_value', 'threshold_value', 'timestamp']
            st.dataframe(alert_df[display_cols], use_container_width=True)
        else:
            st.info("✅ Tidak ada alert aktif")
        
        st.markdown("---")
        
        # Recent Audit Events Section
        st.subheader("📝 Log Audit Terbaru")
        
        recent_events = audit_trail.get_recent_events(20)
        
        if recent_events:
            event_df = pd.DataFrame(recent_events)
            event_df['timestamp'] = pd.to_datetime(event_df['timestamp'])
            
            # Display relevant columns
            display_cols = ['timestamp', 'event_type', 'action', 'resource', 'severity']
            available_cols = [col for col in display_cols if col in event_df.columns]
            
            st.dataframe(event_df[available_cols].head(10), use_container_width=True)
            
            # Event type breakdown
            st.markdown("**Distribusi Event Type:**")
            event_counts = event_df['event_type'].value_counts()
            st.bar_chart(event_counts)
        else:
            st.info("ℹ️ Belum ada event audit tercatat")
        
        st.markdown("---")
        
        # System Information Section
        st.subheader("ℹ️ Informasi Sistem")
        
        try:
            import psutil
            import platform
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**CPU Usage:**")
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("Current", f"{cpu_percent}%")
                
                st.markdown("**Memory Usage:**")
                mem = psutil.virtual_memory()
                st.metric("Used", f"{mem.percent}%")
                st.caption(f"{mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB")
            
            with col2:
                st.markdown("**Disk Usage:**")
                disk = psutil.disk_usage('/')
                st.metric("Used", f"{disk.percent}%")
                st.caption(f"{disk.used / (1024**3):.1f} GB / {disk.total / (1024**3):.1f} GB")
                
                st.markdown("**Platform:**")
                st.metric("OS", platform.system())
                st.caption(f"{platform.release()}")
            
            with col3:
                st.markdown("**Python Version:**")
                st.metric("Version", platform.python_version())
                
                st.markdown("**Uptime:**")
                uptime = time.time() - metrics_collector.performance.metrics.get('started_at', time.time())
                hours = int(uptime // 3600)
                minutes = int((uptime % 3600) // 60)
                st.metric("Duration", f"{hours}h {minutes}m")
            
        except ImportError:
            st.warning("⚠️ psutil tidak tersedia. Install dengan: pip install psutil")
        except Exception as e:
            st.error(f"❌ Error getting system info: {e}")
        
        st.markdown("---")
        
        # Error Context Demo Section
        st.subheader("🔧 Error Context Demo")
        
        st.markdown("Contoh pesan error yang akan ditampilkan ke user:")
        
        error_types = [
            ('MemoryError', 'Memori Tidak Cukup'),
            ('ValueError', 'Nilai Data Tidak Valid'),
            ('FileNotFoundError', 'File Tidak Ditemukan'),
            ('ConnectionError', 'Koneksi Gagal')
        ]
        
        for error_type, title in error_types:
            try:
                raise error_type("Sample error message")
            except Exception as e:
                context = ErrorContext.get_context(e)
                with st.expander(f"💡 {title}"):
                    st.markdown(f"**Tips:**")
                    for tip in context['tips']:
                        st.markdown(tip)
        
        st.markdown("---")
        
        # Quick Actions Section
        st.subheader("⚡ Aksi Cepat")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Bersihkan Cache Lama", use_container_width=True):
                try:
                    cleaned = audit_trail.cleanup_old_logs(days_to_keep=30)
                    st.success(f"✅ {cleaned} file log lama dibersihkan")
                except Exception as e:
                    st.error(f"❌ Gagal membersihkan cache: {e}")
        
        with col2:
            if st.button("📊 Export Metrics", use_container_width=True):
                try:
                    import json
                    status = get_system_status()
                    st.json(status)
                    st.success("✅ Metrics berhasil diekspor")
                except Exception as e:
                    st.error(f"❌ Gagal export metrics: {e}")
        
        with col3:
            if st.button("🔄 Reset Counters", use_container_width=True):
                try:
                    for counter_name in all_metrics['counters']:
                        metrics_collector.performance.reset_counter(counter_name)
                    st.success("✅ Counters berhasil di-reset")
                except Exception as e:
                    st.error(f"❌ Gagal reset counters: {e}")
        
        st.markdown("---")
        
        # Footer
        st.caption("📊 Dashboard Status Sistem ASTINA v2.0 | Enhanced Monitoring & Observability")
        
    except ImportError as e:
        st.error(f"❌ Modul monitoring tidak tersedia: {e}")
        st.info("Pastikan semua dependencies terinstall: pip install -r requirements.txt")
    except Exception as e:
        st.error(f"❌ Error menampilkan status sistem: {e}")
        import traceback
        st.code(traceback.format_exc(), language='python')
