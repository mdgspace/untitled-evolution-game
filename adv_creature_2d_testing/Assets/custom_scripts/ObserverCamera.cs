using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(Camera))]
public class ObserverCamera : MonoBehaviour
{
    public float panSpeed = 28f;
    public float worldMinX = -WorldLayout.WorldBoundaryHalfWidth;
    public float worldMaxX = WorldLayout.WorldBoundaryHalfWidth;
    public float worldMinY = -8f;
    public float worldMaxY = 24f;
    public float minimumZoom = 4f;
    public float maximumZoom = 8f;
    public float zoomSpeed = .02f;
    public bool framePopulationOnStart = false;
    // Matches the older scene framing the user uses as the visual baseline.
    public float startupZoom = 6.87857f;
    private Camera observer;
    private Vector2 lastPointer;
    private bool dragging;
    private int startupFrameAttempts;

    private void Awake() { observer = GetComponent<Camera>(); observer.orthographic = true; }
    private void Start()
    {
        observer.orthographicSize = Mathf.Clamp(startupZoom, minimumZoom, maximumZoom);
        startupFrameAttempts = framePopulationOnStart ? 60 : 0;
    }
    private void Update()
    {
        if (startupFrameAttempts > 0)
        {
            FramePopulation();
            startupFrameAttempts--;
        }
        Keyboard keyboard = Keyboard.current;
        bool left = keyboard != null && (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed);
        bool right = keyboard != null && (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed);
        float horizontal = (right ? 1f : 0f) - (left ? 1f : 0f);
        Vector3 position = transform.position;
        position.x += horizontal * panSpeed * Time.unscaledDeltaTime;

        Mouse mouse = Mouse.current;
        if (mouse != null) {
            Vector2 pointer = mouse.position.ReadValue();
            bool held = mouse.rightButton.isPressed || mouse.middleButton.isPressed;
            if (held && dragging) {
                Vector2 pixels = pointer - lastPointer;
                float unitsPerPixel = observer.orthographicSize * 2f / Mathf.Max(1f, Screen.height);
                position -= new Vector3(pixels.x, pixels.y, 0f) * unitsPerPixel;
            }
            dragging = held; lastPointer = pointer;
            float scroll = mouse.scroll.ReadValue().y;
            if (Mathf.Abs(scroll) > .01f)
                observer.orthographicSize = Mathf.Clamp(observer.orthographicSize - scroll * zoomSpeed, minimumZoom, maximumZoom);
        }
        float halfHeight = observer.orthographicSize;
        float halfWidth = halfHeight * observer.aspect;
        position.x = worldMaxX - worldMinX <= 2f * halfWidth ? (worldMinX + worldMaxX) * .5f : Mathf.Clamp(position.x, worldMinX + halfWidth, worldMaxX - halfWidth);
        position.y = worldMaxY - worldMinY <= 2f * halfHeight ? (worldMinY + worldMaxY) * .5f : Mathf.Clamp(position.y, worldMinY + halfHeight, worldMaxY - halfHeight);
        transform.position = position;
    }

    private void FramePopulation()
    {
        CreatureIdentity[] identities = FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude);
        if (identities.Length == 0) return;
        Bounds bounds = new Bounds(identities[0].transform.position, Vector3.zero);
        foreach (CreatureIdentity identity in identities)
            if (identity != null && identity.torso != null) bounds.Encapsulate(identity.torso.transform.position);
        float aspect = Mathf.Max(.1f, observer.aspect);
        float margin = 8f;
        float requiredSize = Mathf.Max(bounds.extents.y + margin, (bounds.extents.x + margin) / aspect);
        observer.orthographicSize = Mathf.Clamp(requiredSize, minimumZoom, maximumZoom);
        Vector3 position = transform.position;
        position.x = bounds.center.x;
        position.y = bounds.center.y;
        transform.position = position;
    }
}
