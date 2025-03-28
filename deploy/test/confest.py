# conftest.py
def pytest_sessionfinish(session, exitstatus):
    from test_app import success_count, failure_count
    print("\n========== TEST SUMMARY ==========")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failure_count}")
    print("==================================")
    