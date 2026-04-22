def mock_lead_capture(name: str, email: str, platform: str):
    print(f"\n{'='*40}")
    print(f"  LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*40}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{'='*40}\n")
    return "success"