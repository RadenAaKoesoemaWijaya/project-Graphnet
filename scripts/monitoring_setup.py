#!/usr/bin/env python3
"""
ASTINA Production Monitoring Setup

This script helps setup Google Cloud Monitoring dashboards and alerts for ASTINA.
It creates custom metrics, dashboards, and alert policies.

Prerequisites:
- gcloud CLI installed and configured
- Project ID set: gcloud config set project YOUR_PROJECT_ID
- Appropriate IAM permissions (Monitoring Admin)

Usage:
    python scripts/monitoring_setup.py
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd: str, check: bool = True) -> tuple:
    """Run shell command and return (success, output)"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def create_log_based_metric(metric_name: str, filter_expression: str, description: str) -> bool:
    """Create a log-based metric in Cloud Logging"""
    print(f"Creating log-based metric: {metric_name}")
    
    cmd = f"""gcloud logging metrics create {metric_name} \
        --description="{description}" \
        --log-filter='{filter_expression}'"""
    
    success, output = run_command(cmd, check=False)
    
    if success:
        print(f"  ✅ Metric created: {metric_name}")
        return True
    else:
        print(f"  ⚠️  Metric might already exist or failed")
        return True  # Continue even if this fails

def setup_monitoring():
    """Setup Cloud Monitoring for ASTINA"""
    print("=" * 60)
    print("ASTINA Production Monitoring Setup")
    print("=" * 60)
    
    # Get project ID
    success, output = run_command("gcloud config get-value project")
    if success:
        project_id = output.strip()
        if project_id:
            print(f"\nProject ID: {project_id}")
        else:
            print("ERROR: Project ID not set")
            print("Run: gcloud config set project YOUR_PROJECT_ID")
            return False
    else:
        print("ERROR: Could not get project ID")
        return False
    
    print("\n" + "=" * 60)
    print("Creating Log-Based Metrics")
    print("=" * 60)
    
    # Create log-based metrics for ASTINA
    metrics = [
        {
            'name': 'astina_user_login',
            'filter': 'jsonPayload.event_type="USER_LOGIN"',
            'description': 'Count of user login events'
        },
        {
            'name': 'astina_detection_run',
            'filter': 'jsonPayload.event_type="DETECTION_EXECUTION"',
            'description': 'Count of anomaly detection runs'
        },
        {
            'name': 'astina_training_run',
            'filter': 'jsonPayload.event_type="MODEL_TRAINING"',
            'description': 'Count of model training runs'
        },
        {
            'name': 'astina_anomaly_detected',
            'filter': 'jsonPayload.event_type="ANOMALY_DETECTED"',
            'description': 'Count of anomalies detected'
        },
        {
            'name': 'astina_error',
            'filter': 'severity="ERROR"',
            'description': 'Count of error logs'
        },
        {
            'name': 'astina_security_alert',
            'filter': 'jsonPayload.event_type="SECURITY_ALERT"',
            'description': 'Count of security alerts'
        }
    ]
    
    for metric in metrics:
        create_log_based_metric(metric['name'], metric['filter'], metric['description'])
    
    print("\n" + "=" * 60)
    print("Creating Alert Policies")
    print("=" * 60)
    
    # Create alert policy JSON
    alert_policy = {
        "displayName": "ASTINA High Error Rate Alert",
        "documentation": {
            "content": "Alert when error rate exceeds threshold",
            "mimeType": "text/markdown"
        },
        "conditions": [
            {
                "displayName": "Error Rate > 5%",
                "conditionThreshold": {
                    "filter": f'resource.type="cloud_run_revision" AND metric.type="logging.googleapis.com/user/astina_error"',
                    "aggregation": {
                        "alignmentPeriod": "300s",
                        "perSeriesAligner": "ALIGN_RATE",
                        "crossSeriesReducer": "REDUCE_SUM"
                    },
                    "comparison": "COMPARISON_GT",
                    "thresholdValue": 0.05,
                    "duration": "300s"
                }
            }
        ],
        "alertStrategy": {
            "notificationPrompts": [
                {
                    "notificationChannelStrategy": [
                        {
                            "notificationChannelIds": [],
                            "renotificationInterval": "3600s"
                        }
                    ]
                }
            ]
        },
        "enabled": True
    }
    
    # Save alert policy to file
    alert_policy_file = Path("temp_alert_policy.json")
    with open(alert_policy_file, 'w') as f:
        json.dump(alert_policy, f, indent=2)
    
    print("Alert policy configuration saved to temp_alert_policy.json")
    print("To create the alert policy, run:")
    print(f"  gcloud alpha monitoring policies create --policy-from-file={alert_policy_file}")
    
    print("\n" + "=" * 60)
    print("Creating Dashboard Configuration")
    print("=" * 60)
    
    # Create dashboard JSON
    dashboard = {
        "displayName": "ASTINA Production Dashboard",
        "dashboardFilters": [],
        "widgets": [
            {
                "title": "User Logins",
                "xyChart": {
                    "dataSets": [
                        {
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": f'resource.type="cloud_run_revision" AND metric.type="logging.googleapis.com/user/astina_user_login"',
                                    "aggregation": {
                                        "alignmentPeriod": "300s",
                                        "perSeriesAligner": "ALIGN_RATE"
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            {
                "title": "Detection Runs",
                "xyChart": {
                    "dataSets": [
                        {
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": f'resource.type="cloud_run_revision" AND metric.type="logging.googleapis.com/user/astina_detection_run"',
                                    "aggregation": {
                                        "alignmentPeriod": "300s",
                                        "perSeriesAligner": "ALIGN_RATE"
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            {
                "title": "Anomalies Detected",
                "xyChart": {
                    "dataSets": [
                        {
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": f'resource.type="cloud_run_revision" AND metric.type="logging.googleapis.com/user/astina_anomaly_detected"',
                                    "aggregation": {
                                        "alignmentPeriod": "300s",
                                        "perSeriesAligner": "ALIGN_RATE"
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            {
                "title": "Error Rate",
                "xyChart": {
                    "dataSets": [
                        {
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": f'resource.type="cloud_run_revision" AND metric.type="logging.googleapis.com/user/astina_error"',
                                    "aggregation": {
                                        "alignmentPeriod": "300s",
                                        "perSeriesAligner": "ALIGN_RATE"
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    # Save dashboard to file
    dashboard_file = Path("temp_dashboard.json")
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print("Dashboard configuration saved to temp_dashboard.json")
    print("To create the dashboard, run:")
    print(f"  gcloud monitoring dashboards create --config={dashboard_file}")
    
    print("\n" + "=" * 60)
    print("Setting up Uptime Check")
    print("=" * 60)
    
    # Create uptime check
    print("To create uptime check, run:")
    print("  gcloud monitoring uptime-checks create astina-uptime \\")
    print("    --selected-regions=asia-southeast1 \\")
    print("    --resource-type=uptime_url \\")
    print("    --resource-labels=host=YOUR_CLOUD_RUN_URL \\")
    print("    --check-interval=60s \\")
    print("    --timeout=10s \\")
    print("    --content-matchers=content=\"ASTINA\"")
    
    print("\n" + "=" * 60)
    print("✅ MONITORING SETUP COMPLETED")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Review temp_alert_policy.json and create alert policy")
    print("2. Review temp_dashboard.json and create dashboard")
    print("3. Setup uptime check with your Cloud Run URL")
    print("4. Configure notification channels for alerts")
    print("5. Set up log sinks for centralized logging")
    
    return True

def main():
    """Main entry point"""
    try:
        success = setup_monitoring()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()