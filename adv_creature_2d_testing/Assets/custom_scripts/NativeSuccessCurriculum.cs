using System;
using System.Collections.Generic;
using UnityEngine;

public enum NativeTrialMode { Settling, Training, Qualification, Practice, Retention, Complete }

[Serializable]
public sealed class NativeSustainedMovementTrial
{
    public const int WarmupTicks=250,WindowTicks=100,MaximumStationaryTicks=250,RequiredTicks=3000;
    public int elapsedTicks,evaluatedTicks,windowTicks,windowCount,positiveWindows,stationaryTicks,maxStationaryTicks;
    public float startProgress,lastProgress,windowStartProgress,totalProgress,torsoWidth=1f;
    public void Reset(float progress,float width){elapsedTicks=evaluatedTicks=windowTicks=windowCount=positiveWindows=stationaryTicks=maxStationaryTicks=0;startProgress=lastProgress=windowStartProgress=progress;totalProgress=0f;torsoWidth=Mathf.Max(.01f,width);}
    public void Observe(float progress,float fixedDeltaTime){elapsedTicks++;float delta=progress-lastProgress;lastProgress=progress;if(elapsedTicks<=WarmupTicks){startProgress=windowStartProgress=progress;return;}evaluatedTicks++;totalProgress=progress-startProgress;float normalizedSpeed=Mathf.Abs(delta)/(torsoWidth*Mathf.Max(1e-5f,fixedDeltaTime));if(normalizedSpeed<.01f)stationaryTicks++;else stationaryTicks=0;maxStationaryTicks=Mathf.Max(maxStationaryTicks,stationaryTicks);windowTicks++;if(windowTicks>=WindowTicks){windowCount++;if(progress-windowStartProgress>0f)positiveWindows++;windowStartProgress=progress;windowTicks=0;}}
    public float MeanCommandedSpeed(float fixedDeltaTime)=>evaluatedTicks<=0?0f:totalProgress/(torsoWidth*evaluatedTicks*Mathf.Max(1e-5f,fixedDeltaTime));
    public float PositiveWindowFraction=>windowCount<=0?0f:positiveWindows/(float)windowCount;
    public bool Qualified(float fixedDeltaTime)=>elapsedTicks>=RequiredTicks&&MeanCommandedSpeed(fixedDeltaTime)>=.1f&&PositiveWindowFraction>=.8f&&maxStationaryTicks<=MaximumStationaryTicks;
}

[Serializable]
public sealed class NativeCurriculumState
{
    public const int QualificationTarget=8,QualificationTrials=10,AssistedSuccesses=10,RetrySuccesses=5,PracticeInterval=5;
    public int stage,ordinaryTrials,trainingSuccesses,qualificationSuccesses,qualificationAttempts,lifetimeSuccesses,practiceSuccesses,failures,retrySuccesses=RetrySuccesses,retentionChecks,retentionFailures,retentionStage=-1,regressionPracticeTrials;
    public NativeTrialMode mode=NativeTrialMode.Settling;
    public ulong randomState;
    public NativeBrainWeights qualificationSnapshot,validatedSnapshot;
    public NativeCreatureSave qualificationLearningState,validatedLearningState;
    public bool retentionUsingSnapshot;
    public List<int> passedStages=new List<int>();
    public bool Complete=>stage>=NativeSuccessCurriculum.StageCount;
    public float Assistance=>mode==NativeTrialMode.Qualification||mode==NativeTrialMode.Retention||mode==NativeTrialMode.Complete?0f:Mathf.Clamp01(1f-trainingSuccesses/(float)AssistedSuccesses)*NativeCreatureModel.AssistedBiasStrength;
    public bool IsPractice=>mode==NativeTrialMode.Practice||regressionPracticeTrials>0;
    public NativeCurriculumState Clone(){var value=(NativeCurriculumState)MemberwiseClone();value.qualificationSnapshot=qualificationSnapshot?.Clone();value.validatedSnapshot=validatedSnapshot?.Clone();value.qualificationLearningState=qualificationLearningState?.Clone();value.validatedLearningState=validatedLearningState?.Clone();value.passedStages=new List<int>(passedStages??new List<int>());return value;}
}

public static class NativeSuccessCurriculum
{
    public const int StageCount=16;
    private static readonly float[] Distances={.5f,.5f,1f,1f,2f,2f,5f,5f,40f,40f,50f,50f,60f,60f,60f,60f};
    private static readonly float[] Slopes={0f,0f,0f,0f,0f,0f,0f,0f,0f,0f,5f,5f,10f,10f,15f,15f};
    private static readonly int[] Directions={1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1};
    public static float Distance(int stage)=>Distances[Mathf.Clamp(stage,0,StageCount-1)];
    public static float SlopeDegrees(int stage)=>Slopes[Mathf.Clamp(stage,0,StageCount-1)];
    public static int Direction(int stage)=>Directions[Mathf.Clamp(stage,0,StageCount-1)];
    public static float GoalTolerance(int stage){float distance=Distance(stage);return distance<=5f?Mathf.Min(.25f,distance*.2f):.5f;}
    public static string Label(int stage)=>stage>=StageCount?"complete":$"{(Direction(stage)>0?"right":"left")} {Distance(stage):0.##}m {SlopeDegrees(stage):0}deg";
    public static void StartNextTrial(NativeCurriculumState state){if(state.Complete){state.ordinaryTrials++;state.mode=NativeTrialMode.Practice;return;}if(state.mode==NativeTrialMode.Qualification||state.mode==NativeTrialMode.Retention)return;state.ordinaryTrials++;if(state.regressionPracticeTrials>0){state.regressionPracticeTrials--;state.mode=NativeTrialMode.Practice;return;}if(state.passedStages.Count>0&&state.ordinaryTrials%20==0){state.mode=NativeTrialMode.Retention;state.retentionStage=-1;state.retentionUsingSnapshot=false;}else state.mode=state.passedStages.Count>0&&state.ordinaryTrials%NativeCurriculumState.PracticeInterval==0?NativeTrialMode.Practice:NativeTrialMode.Training;}
    public static int TrialStage(NativeCurriculumState state,ref NativeDeterministicRng random){if((!state.IsPractice&&state.mode!=NativeTrialMode.Retention)||state.passedStages.Count==0)return state.stage;if(state.mode==NativeTrialMode.Retention&&state.retentionStage>=0)return state.retentionStage;int stage=state.passedStages[random.NextInt(state.passedStages.Count)];if(state.mode==NativeTrialMode.Retention)state.retentionStage=stage;return stage;}
    public static void RegisterSuccess(NativeCurriculumState state,NativeBrainWeights current){state.lifetimeSuccesses++;if(state.mode==NativeTrialMode.Practice){state.practiceSuccesses++;return;}if(state.mode==NativeTrialMode.Retention){state.retentionChecks++;return;}if(state.mode==NativeTrialMode.Training){state.trainingSuccesses++;state.retrySuccesses=Mathf.Min(NativeCurriculumState.RetrySuccesses,state.retrySuccesses+1);if(state.trainingSuccesses>=NativeCurriculumState.AssistedSuccesses&&state.retrySuccesses>=NativeCurriculumState.RetrySuccesses)BeginQualification(state,current);return;}if(state.mode!=NativeTrialMode.Qualification)return;state.qualificationSuccesses++;state.qualificationAttempts++;FinishQualificationIfReady(state,current);}
    public static void RegisterFailure(NativeCurriculumState state,NativeBrainWeights current){state.failures++;if(state.mode==NativeTrialMode.Retention){state.retentionFailures++;return;}if(state.mode!=NativeTrialMode.Qualification)return;state.qualificationAttempts++;FinishQualificationIfReady(state,current);}
    public static void FlagRegression(NativeCurriculumState state){if(state==null)return;state.regressionPracticeTrials=Mathf.Max(state.regressionPracticeTrials,5);if(state.passedStages.Count>0){int revoked=state.passedStages[state.passedStages.Count-1];state.passedStages.RemoveAll(s=>s>=revoked);state.stage=Mathf.Min(state.stage,revoked);}}
    private static void BeginQualification(NativeCurriculumState state,NativeBrainWeights current){state.mode=NativeTrialMode.Qualification;state.qualificationAttempts=state.qualificationSuccesses=0;state.qualificationSnapshot=current.Clone();}
    private static void FinishQualificationIfReady(NativeCurriculumState state,NativeBrainWeights current){if(state.qualificationAttempts<NativeCurriculumState.QualificationTrials)return;if(state.qualificationSuccesses>=NativeCurriculumState.QualificationTarget){state.passedStages.Add(state.stage);state.validatedSnapshot=(state.qualificationSnapshot??current).Clone();state.stage++;state.trainingSuccesses=0;state.retrySuccesses=NativeCurriculumState.RetrySuccesses;state.qualificationAttempts=state.qualificationSuccesses=0;state.mode=state.Complete?NativeTrialMode.Complete:NativeTrialMode.Training;}else{state.mode=NativeTrialMode.Training;state.retrySuccesses=0;state.qualificationAttempts=state.qualificationSuccesses=0;state.qualificationSnapshot=null;state.qualificationLearningState=null;}}
}
