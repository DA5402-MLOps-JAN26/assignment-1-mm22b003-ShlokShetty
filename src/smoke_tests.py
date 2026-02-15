import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
LOG_FILE = "smoke_test_log.txt"


def log(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def test_health_check():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "active_model_version" in data
    assert "threshold" in data

    print("Health check test passed")
    log("test_health_check: PASS")


def test_1():
    payload = {
        "Type": 1,
        "Air_temperature_K": 300,
        "Process_temperature_K": 310,
        "Rotational_speed_rpm": 1500,
        "Torque_Nm": 40,
        "Tool_wear_min": 120
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "model_version" in data
    assert "failure_probability" in data
    assert "threshold" in data
    assert "prediction" in data
    assert data["prediction"] in [0, 1]

    
    # log(
    #     f"test_result_1: PASS | "
    #     f"prediction={data['prediction']} prob={data['failure_probability']}"
    # )
    print(f"test_result_1: PASS | "f"prediction={data['prediction']} prob={data['failure_probability']}")


def test_2():
    payload = {
        "Type": 2,
        "Air_temperature_K": 315,
        "Process_temperature_K": 330,
        "Rotational_speed_rpm": 2500,
        "Torque_Nm": 80,
        "Tool_wear_min": 250
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["prediction"] in [0, 1]
    
    
    # log(
    #     f"test_result_2: PASS | "
    #     f"prediction={data['prediction']} prob={data['failure_probability']}"
    # )
    print(f"test_result_2: PASS | "f"prediction={data['prediction']} prob={data['failure_probability']}")


def test_3():
    payload = {
        "Type": 1,
        "Air_temperature_K": 305,
        "Process_temperature_K": 320,
        "Rotational_speed_rpm": 1800,
        "Torque_Nm": 55,
        "Tool_wear_min": 180
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["prediction"] in [0, 1]

    
    # log(
    #     f"test_result_3: PASS | "
    #     f"prediction={data['prediction']} prob={data['failure_probability']}"
    # )
    print(f"test_result_3: PASS | "f"prediction={data['prediction']} prob={data['failure_probability']}")



if __name__ == "__main__":
    try:
        test_health_check()
        test_1()
        test_2()
        test_3()

        print("Passed")

    except AssertionError as e:
        print("Smoke test failed")
        log(f"SMOKE TEST FAILED: {e}")
        raise
