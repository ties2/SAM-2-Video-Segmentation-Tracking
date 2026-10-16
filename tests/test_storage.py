import pytest

from src.serving.storage import ArtifactNotFoundError, InvalidURIError, LocalArtifactStorage, PathTraversalError


# Create a temporary Storage for running tests
@pytest.fixture
def storage(tmp_path):
    # tmp_path creates a secure, temporary directory
    return LocalArtifactStorage(base_dir=str(tmp_path))


def test_save_and_get_artifact(storage):
    """Test 1 & 2: Check successful saving and retrieval of a file"""
    uri = "local://fraud-model/1.0.0/model.joblib"
    content = b"dummy_model_binary_content"

    # Save the file
    storage.save(uri, content)

    # Check the existence and content of the file
    assert storage.exists(uri) is True
    assert storage.get(uri) == content


def test_exists_returns_false_for_missing_artifact(storage):
    """Test 3 & 4: Check the exists method for existing and missing files"""
    assert storage.exists("local://missing/model.pkl") is False


def test_get_missing_artifact_raises_error(storage):
    """Check that the correct error is raised when retrieving a missing file"""
    with pytest.raises(ArtifactNotFoundError):
        storage.get("local://missing/model.pkl")


def test_delete_artifact(storage):
    """Test 5: Check file deletion functionality"""
    uri = "local://fraud-model/delete-test/model.joblib"
    storage.save(uri, b"data_to_delete")

    assert storage.exists(uri) is True

    # Delete the file
    storage.delete(uri)
    assert storage.exists(uri) is False


def test_invalid_uri_rejected(storage):
    """Test 6: Check rejection of non-local URIs"""
    with pytest.raises(InvalidURIError):
        storage.save("s3://my-bucket/model.joblib", b"data")

    with pytest.raises(InvalidURIError):
        storage.get("s3://my-bucket/model.joblib")


def test_path_traversal_rejected(storage):
    """Test 7: Check prevention of Path Traversal attacks"""
    # Attempt to escape the base directory using ../
    malicious_uri = "local://../../../etc/passwd"

    with pytest.raises(PathTraversalError):
        storage.save(malicious_uri, b"malicious_data")

    with pytest.raises(PathTraversalError):
        storage.get(malicious_uri)
