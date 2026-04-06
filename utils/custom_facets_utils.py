import requests


def get_custom_facets(ikigai, headers):
    """Get custom facets for a user.
    Args:
        ikigai (Ikigai): Ikigai client instance.
        headers (dict): Headers for the request including authentication details.

    """
    BASE_URL = ikigai.base_url

    url = f"{BASE_URL}/component/get-custom-facets-for-user"
    response = requests.request("GET", url, headers=headers, data={}, verify=False)
    cf = response.json()["custom_facets"]

    return cf


def get_custom_facet_by_name(ikigai, headers, facet_name, fail_if_not_found=True):
    """Get a custom facet by name.
    Args:
       ikigai (Ikigai): Ikigai client instance.
       headers (dict): Headers for the request including authentication details.
       facet_name (str): Name of the custom facet to retrieve.

    Returns:
        dict: Custom facet details if found, else None.
    """
    custom_facets = get_custom_facets(ikigai, headers)
    for facet in custom_facets:
        if facet["name"] == facet_name:
            return facet
    if fail_if_not_found:
        raise ValueError(f"Custom facet with name '{facet_name}' not found.")
    return None


def create_custom_facet(
    ikigai,
    headers,
    facet_name,
    script,
    chain="MID",
    libraries=None,
    arg_list=None,
    description="test",
):
    """Create a custom facet."""
    BASE_URL = ikigai.base_url
    if libraries is None:
        libraries = []
    if arg_list is None:
        arg_list = []
    url = f"{BASE_URL}/component/create-custom-facet"

    payload = {
        "custom_facet": {
            "name": facet_name,
            "chain_group": chain,
            "python_script": script,
            "libraries": libraries,
            "rootkit_token": "",
            "arguments": arg_list,
            "description": description,
        }
    }
    response = requests.request(
        "POST", url, headers=headers, json=payload, verify=False
    )
    return response.json()


def edit_custom_facet(
    ikigai,
    headers,
    facet_id,
    facet_name,
    script,
    chain="MID",
    libraries=None,
    arg_list=None,
    description="test",
):
    """Edit a custom facet."""
    BASE_URL = ikigai.base_url
    if libraries is None:
        libraries = []
    if arg_list is None:
        arg_list = []
    url = f"{BASE_URL}/component/edit-custom-facet"

    payload = {
        "custom_facet": {
            "custom_facet_id": facet_id,
            "name": facet_name,
            "chain_group": chain,
            "python_script": script,
            "libraries": libraries,
            "rootkit_token": "",
            "arguments": arg_list,
            "description": description,
        }
    }
    response = requests.request(
        "POST", url, headers=headers, json=payload, verify=False
    )
    return response.json()


def delete_custom_facet(ikigai, headers, facet_id):
    """Delete a custom facet."""
    BASE_URL = ikigai.base_url

    url = f"{BASE_URL}/component/delete-custom-facet"
    payload = {"custom_facet_id": facet_id}
    response = requests.request(
        "POST", url, headers=headers, json=payload, verify=False
    )
    return response.json()


SHARE_EMAILS = [
    "abdullah@ikigailabs.io",
    "adharask@ikigailabs.io",
    "nathaniel@ikigailabs.io",
]


def deploy_custom_facet(ikigai, facet_name, script, description="", share_emails=None):
    """Deploy a custom facet by name using the Ikigai SDK.

    If a custom facet with the given name already exists it is deleted first,
    then a fresh one is created with the supplied script.  This guarantees
    the script is always present and that .unpinned() resolves correctly
    (update_script only touches the draft; a new version would be needed
    for unpinned to pick it up — deleting + recreating avoids that issue).

    After creation the facet is automatically shared (ADMIN) with every
    address in ``share_emails`` (defaults to SHARE_EMAILS).

    Returns:
        CustomFacet: The custom facet object (call .unpinned() to use in flows).
    """
    from ikigai import CustomFacetAccessLevel

    if share_emails is None:
        share_emails = SHARE_EMAILS

    # Delete existing facet if present
    try:
        existing = ikigai.custom_facets[facet_name]
        print(f"  Deleting existing custom facet '{facet_name}' …")
        existing.delete()
    except Exception:
        pass  # facet does not exist yet

    # Create fresh
    print(f"  Creating custom facet '{facet_name}' …")
    cf = (
        ikigai.custom_facet.new(
            name=facet_name,
            facet_type=ikigai.facet_types.MID.CUSTOM_FACET,
        )
        .script(script=script)
        .description(description)
        .build()
    )

    # Share with configured emails
    for email in share_emails:
        try:
            cf.access.grant(email=email, access_level=CustomFacetAccessLevel.ADMIN)
        except Exception as exc:
            print(f"  ⚠ Could not share '{facet_name}' with {email}: {exc}")
    if share_emails:
        print(f"  Shared '{facet_name}' (ADMIN) with {len(share_emails)} user(s)")

    return cf


def share_custom_facet(
    ikigai, headers, custom_facet_id, target_user_email, access_level="READ"
):
    """Share a custom facet with another user.

    Args:
        ikigai (Ikigai): Ikigai client instance.
        headers (dict): Headers for the request including authentication details.
        custom_facet_id (str): ID of the custom facet to share.
        target_user_email (str): Email of the user to share with.
        access_level (str): Access level for the shared facet (default: "READ").

    Returns:
        dict: API response.
    """
    BASE_URL = ikigai.base_url

    url = f"{BASE_URL}/component/share-custom-facet"
    payload = {
        "custom_facet": {"custom_facet_id": custom_facet_id},
        "user": {"email": target_user_email},
        "access_level": access_level,
    }
    response = requests.request(
        "POST", url, headers=headers, json=payload, verify=False
    )
    return response.json()
