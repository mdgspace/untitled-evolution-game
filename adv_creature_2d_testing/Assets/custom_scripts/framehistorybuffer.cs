using System.Collections.Generic;

// Generic sliding-window memory buffer for the goal-setter's inputs.
// Holds the last `frameCount` frames of a fixed-size float vector (used
// for both position history and vision-sighting history below -- same
// class, different vector sizes) and hands back the whole window
// concatenated into one flat array, oldest frame first.
//
// On the very first PushFrame call, every slot gets filled with that same
// first frame rather than starting from zeros -- per spec, so early-life
// inputs don't look like a discontinuous jump from nothing.
public class FrameHistoryBuffer
{
    private readonly int frameCount;
    private readonly int vectorSize;
    private readonly List<float[]> frames;
    private bool initialized = false;

    public FrameHistoryBuffer(int frameCount, int vectorSize)
    {
        this.frameCount = frameCount;
        this.vectorSize = vectorSize;
        frames = new List<float[]>(frameCount);
    }

    public void PushFrame(float[] currentFrame)
    {
        if (currentFrame.Length != vectorSize)
        {
            throw new System.ArgumentException(
                $"FrameHistoryBuffer expected a vector of size {vectorSize}, got {currentFrame.Length}");
        }

        if (!initialized)
        {
            for (int i = 0; i < frameCount; i++)
                frames.Add((float[])currentFrame.Clone());
            initialized = true;
            return;
        }

        frames.Add((float[])currentFrame.Clone());
        if (frames.Count > frameCount)
            frames.RemoveAt(0);
    }

    public float[] GetConcatenated()
    {
        float[] result = new float[frameCount * vectorSize];
        for (int i = 0; i < frames.Count; i++)
            System.Array.Copy(frames[i], 0, result, i * vectorSize, vectorSize);
        return result;
    }
}