using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(Camera))]
public class ObserverCamera : MonoBehaviour
{
    public float panSpeed = 28f;
    public float worldMinX = -130f;
    public float worldMaxX = 130f;
    public float worldMinY = -8f;
    public float worldMaxY = 24f;
    public float minimumZoom = 4f;
    public float maximumZoom = 14f;
    public float zoomSpeed = .02f;
    private Camera observer;
    private Vector2 lastPointer;
    private bool dragging;

    private void Awake() { observer = GetComponent<Camera>(); observer.orthographic = true; }
    private void Update()
    {
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
        position.x = Mathf.Clamp(position.x, worldMinX + halfWidth, worldMaxX - halfWidth);
        position.y = Mathf.Clamp(position.y, worldMinY + halfHeight, worldMaxY - halfHeight);
        transform.position = position;
    }
}
