using UnityEngine;
using UnityEditor;

internal class CommandScript : IRunCommand
{
    public void Execute(ExecutionResult result)
    {
        var n = Object.FindAnyObjectByType<NativeEcosystemController>();
        if (n == null) { result.Log("no controller; playing={0}", EditorApplication.isPlaying); return; }
        string message = "playing=" + EditorApplication.isPlaying + " scale=" + Time.timeScale.ToString("F1") + " fixed=" + Time.fixedDeltaTime.ToString("F3") + " physicsHz=" + n.PhysicsRateHz.ToString("F1") + " tick=" + n.Tick + " pop=" + n.Population + " moving=" + n.MovingCreatureCount + " fps=" + n.RenderRateHz.ToString("F1") + " p95=" + n.FrameP95Milliseconds.ToString("F1") + " bias=" + n.BiasStrength.ToString("F3") + " noise=" + n.ExplorationSigma.ToString("F3") + " action=" + n.LastActionMagnitude.ToString("F3") + " root=" + n.MeanRootSpeed.ToString("F3") + " joint=" + n.MeanJointSpeed.ToString("F3") + " fresh=" + n.FreshSequenceSamples + " exposures=" + n.SharedM3Samples + " m3steps=" + n.SharedM3OptimizerSteps + " m2warm=" + n.M2WarmupComplete + " queue=" + n.LearnerQueueDepth + " drops=" + n.LearnerQueueDrops + " faults=" + n.ControlFailures + " config=" + NativeEcosystemCheckpoint.ConfigurationFingerprint + " status=" + n.Status + " checkpoint=" + n.CheckpointStatus;
        result.Log(message);
    }
}
