import nltk


def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        pass

    error_message = (
        f"Required NLTK resource '{package_name}' is not available locally. "
        f"Tried to resolve '{resource_path}' and attempted an automatic download, but it failed. "
        "Download it in advance on a machine with network access and set NLTK_DATA to the cached directory."
    )

    try:
        downloaded = nltk.download(package_name, quiet=True)
    except Exception as exc:
        raise RuntimeError(error_message) from exc

    if not downloaded:
        raise RuntimeError(error_message)

    try:
        nltk.data.find(resource_path)
    except LookupError as exc:
        raise RuntimeError(error_message) from exc
