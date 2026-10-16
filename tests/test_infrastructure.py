import json
import os

import yaml

# ==========================================
# 8 Automated Tests for Infrastructure
# ==========================================


def test_prometheus_config_exists():
    """Test 1: Prometheus config exists"""
    assert os.path.exists("monitoring/prometheus/prometheus.yml")


def test_grafana_datasource_provisioning_exists():
    """Test 2: Grafana datasource provisioning exists"""
    assert os.path.exists("monitoring/grafana/provisioning/datasources/prometheus.yml")


def test_grafana_dashboard_provisioning_exists():
    """Test 3: Grafana dashboard provisioning exists"""
    assert os.path.exists("monitoring/grafana/provisioning/dashboards/dashboards.yml")


def test_dashboard_json_exists_and_valid():
    """Test 4 & 5: Dashboard JSON exists and has valid panels/variables"""
    dashboard_path = "monitoring/grafana/dashboards/model-serving.json"
    assert os.path.exists(dashboard_path)

    with open(dashboard_path) as f:
        dashboard = json.load(f)
        assert "panels" in dashboard
        assert len(dashboard["panels"]) >= 5
        assert "templating" in dashboard


def test_docker_compose_has_monitoring_services():
    """Test 6 & 7: Compose file has Prometheus and Grafana with volumes"""
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})
    assert "prometheus" in services
    assert "grafana" in services

    volumes = compose.get("volumes", {})
    assert "prometheus_data" in volumes
    assert "grafana_data" in volumes


def test_env_example_has_grafana_credentials():
    """Test 8: Settings model accepts Grafana keys (Pydantic validation)"""
    from src.serving.config import settings

    assert hasattr(settings, "gf_security_admin_user")
    assert hasattr(settings, "gf_security_admin_password")
