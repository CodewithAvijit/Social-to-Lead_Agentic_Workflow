def mock_lead_capture(name: str, email: str, platform: str):
    """
    Mock API function to capture a qualified lead.
    In production, this would POST to a CRM like HubSpot, Salesforce, etc.
    """
    print(f"\n{'='*40}")
    print(f"  LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*40}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{'='*40}\n")
    return "success"