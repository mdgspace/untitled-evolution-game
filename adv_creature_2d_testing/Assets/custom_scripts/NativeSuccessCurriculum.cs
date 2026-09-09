using System;
using System.Collections.Generic;
using UnityEngine;

public enum NativeTrialMode { Settling, Training, Qualification, Practice, Retention, Complete }

[Serializable]
public sealed class NativeCurriculumState
{
    public const int QualificationTarget = 10, QualificationTrials = 12, AssistedSuccesses = 10, RetrySuccesses = 5, PracticeInterval = 5;
    public int stage, ordinaryTrials, trainingSuccesses, qualificationSuccesses, qualificationAttempts, lifetimeSuccesses, practiceSuccesses, failures, retrySuccesses=RetrySuccesses, retentionChecks, retentionFailures,retentionStage=-1;
    public NativeTrialMode mode = NativeTrialMode.Settling;
    public ulong randomState;
    public NativeBrainWeights qualificationSnapshot, validatedSnapshot;
    public NativeCreatureSave qualificationLearningState, validatedLearningState;
    public bool retentionUsingSnapshot;
    public List<int> passedStages = new List<int>();

    public bool Complete => stage >= NativeSuccessCurriculum.StageCount;
    public float Assistance => mode == NativeTrialMode.Qualification || mode == NativeTrialMode.Retention || mode == NativeTrialMode.Complete ? 0f : Mathf.Clamp01(1f - trainingSuccesses / (float)AssistedSuccesses) * .75f;
    public bool IsPractice => mode == NativeTrialMode.Practice;
    public NativeCurriculumState Clone()
    {
        var value = (NativeCurriculumState)MemberwiseClone(); value.qualificationSnapshot = qualificationSnapshot?.Clone(); value.validatedSnapshot = validatedSnapshot?.Clone(); value.qualificationLearningState=qualificationLearningState?.Clone();value.validatedLearningState=validatedLearningState?.Clone();value.passedStages = new List<int>(passedStages??new List<int>()); return value;
    }
}

public static class NativeSuccessCurriculum
{
    public const int StageCount = 8;
    private static readonly float[] Slopes = { 0f, 0f, 5f, 5f, 10f, 10f, 15f, 15f };
    private static readonly int[] Directions = { 1, -1, 1, -1, 1, -1, 1, -1 };
    private static readonly float[] Distances = { 4f, 4f, 5f, 5f, 6f, 6f, 6f, 6f };

    public static float SlopeDegrees(int stage) => Slopes[Mathf.Clamp(stage, 0, StageCount - 1)];
    public static int Direction(int stage) => Directions[Mathf.Clamp(stage, 0, StageCount - 1)];
    public static float Distance(int stage) => Distances[Mathf.Clamp(stage, 0, StageCount - 1)];
    public static string Label(int stage) => stage >= StageCount ? "complete" : $"{(Direction(stage) > 0 ? "right" : "left")} {SlopeDegrees(stage):F0}deg";

    public static void StartNextTrial(NativeCurriculumState state)
    {
        if (state.Complete) { state.ordinaryTrials++;state.mode=NativeTrialMode.Practice;return; }
        if (state.mode == NativeTrialMode.Qualification || state.mode==NativeTrialMode.Retention) return;
        state.ordinaryTrials++;
        if(state.passedStages.Count>0&&state.ordinaryTrials%20==0){state.mode=NativeTrialMode.Retention;state.retentionStage=-1;state.retentionUsingSnapshot=false;}
        else state.mode = state.passedStages.Count > 0 && state.ordinaryTrials % NativeCurriculumState.PracticeInterval == 0 ? NativeTrialMode.Practice : NativeTrialMode.Training;
    }

    public static int TrialStage(NativeCurriculumState state, ref NativeDeterministicRng random)
    {
        if (!state.IsPractice && state.mode!=NativeTrialMode.Retention || state.passedStages.Count == 0) return state.stage;
        if(state.mode==NativeTrialMode.Retention&&state.retentionStage>=0)return state.retentionStage;
        int stage=state.passedStages[random.NextInt(state.passedStages.Count)];if(state.mode==NativeTrialMode.Retention)state.retentionStage=stage;return stage;
    }

    public static void RegisterSuccess(NativeCurriculumState state, NativeBrainWeights current)
    {
        state.lifetimeSuccesses++;
        if (state.mode == NativeTrialMode.Practice) { state.practiceSuccesses++; return; }
        if(state.mode==NativeTrialMode.Retention){state.retentionChecks++;return;}
        if (state.mode == NativeTrialMode.Training)
        {
            state.trainingSuccesses++; state.retrySuccesses=Mathf.Min(NativeCurriculumState.RetrySuccesses,state.retrySuccesses+1);
            if (state.trainingSuccesses >= NativeCurriculumState.AssistedSuccesses && state.retrySuccesses >= NativeCurriculumState.RetrySuccesses) BeginQualification(state, current);
            return;
        }
        if (state.mode != NativeTrialMode.Qualification) return;
        state.qualificationSuccesses++;
        state.qualificationAttempts++;
        FinishQualificationIfReady(state, current);
    }

    public static void RegisterFailure(NativeCurriculumState state, NativeBrainWeights current)
    {
        state.failures++;if(state.mode==NativeTrialMode.Retention){state.retentionFailures++;return;}
        if (state.mode != NativeTrialMode.Qualification) return;
        state.qualificationAttempts++;
        FinishQualificationIfReady(state, current);
    }

    private static void BeginQualification(NativeCurriculumState state, NativeBrainWeights current)
    {
        state.mode = NativeTrialMode.Qualification; state.qualificationAttempts = 0; state.qualificationSuccesses = 0; state.qualificationSnapshot = current.Clone();
    }

    private static void FinishQualificationIfReady(NativeCurriculumState state, NativeBrainWeights current)
    {
        if (state.qualificationAttempts < NativeCurriculumState.QualificationTrials) return;
        if (state.qualificationSuccesses >= NativeCurriculumState.QualificationTarget)
        {
            state.passedStages.Add(state.stage); state.validatedSnapshot = (state.qualificationSnapshot ?? current).Clone(); state.stage++; state.trainingSuccesses = 0; state.retrySuccesses = NativeCurriculumState.RetrySuccesses; state.qualificationAttempts = state.qualificationSuccesses = 0; state.mode = state.Complete ? NativeTrialMode.Complete : NativeTrialMode.Training;
        }
        else
        {
            state.mode = NativeTrialMode.Training; state.retrySuccesses = 0; state.qualificationAttempts = state.qualificationSuccesses = 0; state.qualificationSnapshot = null;state.qualificationLearningState=null;
        }
    }

}
