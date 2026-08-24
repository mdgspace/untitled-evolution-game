using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

[Serializable]
public sealed class NativeBrainWeights
{
    public int version = 1;
    public float[] m1Input;
    public float[] m1Hidden;
    public float[] m1Output;
    public float[] m2Input;
    public float[] m2Middle;
    public float[] m2Deep;
    public float[] m2Extra;
    public float[] m2History;
    public float[] m2Output;
    public float[] m3Input;
    public float[] m3Middle;
    public float[] m3History;
    public float[] m3Output;
    public float[] m2Moment;
    public float[] m3Moment;

    public static NativeBrainWeights Create(int seed)
    {
        System.Random random = new System.Random(seed);
        NativeBrainWeights w = new NativeBrainWeights {
            m1Input = Create(random, 28 * 16 * 3), m1Hidden = Create(random, 16 * 16 * 3), m1Output = Create(random, 16 * 2),
            m2Input = Create(random, 21 * 16), m2Middle = Create(random, 32 * 16), m2Deep = Create(random, 16 * 16), m2Extra = Create(random, 16 * 16), m2History = Create(random, 16 * 16 * 3), m2Output = Create(random, 16 * 2),
            m3Input = Create(random, 13 * 16), m3Middle = Create(random, 16 * 16), m3History = Create(random, 16 * 16 * 3), m3Output = Create(random, 16 * 2), m2Moment = new float[16 * 2], m3Moment = new float[16 * 2]
        };
        return w;
    }

    private static float[] Create(System.Random random, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < values.Length; i++) values[i] = (float)((random.NextDouble() - .5) * .15);
        return values;
    }

    public NativeBrainWeights Clone() => JsonConvert.DeserializeObject<NativeBrainWeights>(JsonConvert.SerializeObject(this));
    public void CopyFrom(NativeBrainWeights source) { m1Input = (float[])source.m1Input.Clone(); m1Hidden = (float[])source.m1Hidden.Clone(); m1Output = (float[])source.m1Output.Clone(); m2Input = (float[])source.m2Input.Clone(); m2Middle = (float[])source.m2Middle.Clone(); m2Deep = (float[])source.m2Deep.Clone(); m2Extra = (float[])source.m2Extra.Clone(); m2History = (float[])source.m2History.Clone(); m2Output = (float[])source.m2Output.Clone(); m3Input = (float[])source.m3Input.Clone(); m3Middle = (float[])source.m3Middle.Clone(); m3History = (float[])source.m3History.Clone(); m3Output = (float[])source.m3Output.Clone(); m2Moment = source.m2Moment == null ? new float[32] : (float[])source.m2Moment.Clone(); m3Moment = source.m3Moment == null ? new float[32] : (float[])source.m3Moment.Clone(); }
    public void Mutate(System.Random random, float amount = .05f)
    {
        foreach (float[] array in Arrays()) for (int i = 0; i < array.Length; i++) if (random.NextDouble() < .08) array[i] += (float)((random.NextDouble() - .5) * amount);
    }
    public void CopyM3From(NativeBrainWeights source)
    {
        Array.Copy(source.m3Input, m3Input, m3Input.Length); Array.Copy(source.m3Middle, m3Middle, m3Middle.Length);
        Array.Copy(source.m3History, m3History, m3History.Length); Array.Copy(source.m3Output, m3Output, m3Output.Length);
        if (m3Moment == null) m3Moment = new float[32]; Array.Copy(source.m3Moment ?? Array.Empty<float>(), m3Moment, Mathf.Min(m3Moment.Length, source.m3Moment?.Length ?? 0));
    }
    public bool IsFinite() => Arrays().All(array => array != null && array.All(value => !float.IsNaN(value) && !float.IsInfinity(value)));
    public IEnumerable<float[]> Arrays() { yield return m1Input; yield return m1Hidden; yield return m1Output; yield return m2Input; yield return m2Middle; yield return m2Deep; yield return m2Extra; yield return m2History; yield return m2Output; yield return m3Input; yield return m3Middle; yield return m3History; yield return m3Output; }
}

public sealed class NativeCreatureModel
{
    public const int History = 4, MaxLimbs = 8, MaxPathDepth = 4, Hidden = 16;
    public const int M2InputSize = 21;
    public static readonly int[] M2PhasePeriods = { 8, 16, 32, 64 };
    public const float MinimumActionDegrees = .75f;
    public const float MaximumActionDegrees = 8f;
    // M1 begins close to zero, so dividing by the instantaneous goal magnitude
    // turned harmless early prediction errors into five-digit losses.  Goals are
    // bounded to 0.25 world units; use that fixed scale for a comparable loss.
    public const float GoalNormalizationSquared = .25f * .25f;
    public const int MinimumM2FitnessSamples = 50;
    public readonly NativeBrainWeights Weights;
    public readonly float[] M1Hidden = new float[Hidden];
    public readonly float[] M2Hidden = new float[Hidden];
    public readonly float[] M3Hidden = new float[Hidden];
    private readonly float[][] frameHistory = { new float[10], new float[10], new float[10], new float[10] };
    private int historyCount;
    private readonly float[] lastM2Combined = new float[Hidden];
    private readonly float[] lastM3Combined = new float[Hidden];
    private readonly System.Random exploration;
    private float lastAction;
    private Vector2 lastPrediction;
    private Vector2 lastTarget;
    private bool hasPrediction;
    private readonly List<NativeDynamicsReplaySample> dynamicsReplay = new List<NativeDynamicsReplaySample>();
    private readonly List<NativePolicyReplaySample> policyReplay = new List<NativePolicyReplaySample>();
    public float LastM2Loss { get; private set; } = float.PositiveInfinity;
    public float LastM3Loss { get; private set; } = float.PositiveInfinity;
    public int M2LossSamples { get; private set; }
    public float M2LossSum { get; private set; }
    public int DynamicsReplayCount => dynamicsReplay.Count;
    public float BodyFitness => M2LossSamples < MinimumM2FitnessSamples ? float.PositiveInfinity : M2LossSum / M2LossSamples;
    public Vector2 CurrentGoal { get; private set; }

    public NativeCreatureModel(NativeBrainWeights weights, int seed = 0) { Weights = weights ?? NativeBrainWeights.Create(Environment.TickCount); exploration = new System.Random(seed == 0 ? Environment.TickCount : seed); }

    public float[] Infer(float[] global, Limb[] limbs, Vector2 goal, int controlStep)
    {
        float[] m1 = RunM1(global);
        Vector2 proposedGoal = new Vector2(m1[0], m1[1]) * .25f;
        CurrentGoal = proposedGoal;
        int limbCount = Mathf.Min(limbs.Length, MaxLimbs);
        float[] actions = new float[MaxLimbs];
        float[][] m2Base = new float[limbCount][];
        float[] gruInput = new float[Hidden];
        for (int i = 0; i < limbCount; i++)
        {
            // The M2 GRU is intentionally evaluated before the M2 stack. Its
            // state produces a history token for the current decision.
            m2Base[i] = Dense(M2LimbFeatures(limbs[i], proposedGoal, controlStep), Weights.m2Input, M2InputSize, Hidden, true);
            for (int h = 0; h < Hidden; h++) gruInput[h] += m2Base[i][h];
        }
        if (limbCount > 0) for (int h = 0; h < Hidden; h++) gruInput[h] /= limbCount;
        float[] m2History = RunGru(gruInput, M2Hidden, Weights.m2History);
        float[][] m2Encoded = new float[limbCount][];
        float[] context = new float[Hidden];
        for (int i = 0; i < limbCount; i++)
        {
            float[] historyInput = new float[Hidden * 2];
            Array.Copy(m2Base[i], 0, historyInput, 0, Hidden); Array.Copy(m2History, 0, historyInput, Hidden, Hidden);
            m2Encoded[i] = Dense(Dense(Dense(historyInput, Weights.m2Middle, Hidden * 2, Hidden, true), Weights.m2Deep, Hidden, Hidden, true), Weights.m2Extra, Hidden, Hidden, true);
            for (int h = 0; h < Hidden; h++) context[h] += m2Encoded[i][h];
        }
        if (limbCount > 0) for (int h = 0; h < Hidden; h++) context[h] /= limbCount;
        for (int i = 0; i < limbCount; i++)
        {
            float sampledRaw = DenseWithContextScalar(m2Encoded[i], context, m2History, Weights.m2Output) + NextGaussian() * .35f;
            actions[i] = BoundedAction(sampledRaw);
            if (i == 0) for (int h = 0; h < Hidden; h++) lastM2Combined[h] = (float)Math.Tanh(m2Encoded[i][h] + context[h] + m2History[h]);
        }
        float predictedX = 0f, predictedY = 0f; Array.Clear(lastM3Combined, 0, lastM3Combined.Length);
        float[][] m3Encoded = new float[limbCount][];
        float[] m3Context = new float[Hidden];
        for (int i = 0; i < limbCount; i++)
        {
            m3Encoded[i] = Dense(Dense(BaseLimbFeatures(limbs[i]), Weights.m3Input, 13, Hidden, true), Weights.m3Middle, Hidden, Hidden, true);
            for (int h = 0; h < Hidden; h++) m3Context[h] += m3Encoded[i][h];
        }
        if (limbCount > 0) for (int h = 0; h < Hidden; h++) m3Context[h] /= limbCount;
        float[] m3History = RunGru(m3Context, M3Hidden, Weights.m3History);
        for (int i = 0; i < limbCount; i++)
        {
            float[] combined = new float[Hidden]; for (int h = 0; h < Hidden; h++) combined[h] = (float)Math.Tanh(m3Encoded[i][h] + m3Context[h] + m3History[h]); for (int h = 0; h < Hidden; h++) lastM3Combined[h] += combined[h];
            float[] output = DenseWithContextVector(m3Encoded[i], m3Context, m3History, Weights.m3Output, actions[i]);
            predictedX += output[0]; predictedY += output[1];
        }
        if (limbCount > 0) { predictedX /= limbCount; predictedY /= limbCount; for (int h = 0; h < Hidden; h++) lastM3Combined[h] /= limbCount; }
        LastM2Loss = ((predictedX - proposedGoal.x) * (predictedX - proposedGoal.x) + (predictedY - proposedGoal.y) * (predictedY - proposedGoal.y)) / GoalNormalizationSquared;
        lastPrediction = new Vector2(predictedX, predictedY); lastTarget = proposedGoal; lastAction = actions.Length == 0 ? 0f : actions[0]; hasPrediction = limbCount > 0;
        M2LossSum += LastM2Loss; M2LossSamples++;
        policyReplay.Add(new NativePolicyReplaySample { combined = (float[])lastM2Combined.Clone(), predictionX = lastPrediction.x, predictionY = lastPrediction.y, targetX = lastTarget.x, targetY = lastTarget.y });
        if (policyReplay.Count > 128) policyReplay.RemoveAt(0);
        return actions;
    }

    // Bounded manual backward pass for the recurrent network's output layers.
    // The fixed-size kernels keep training allocations out of Unity's frame loop.
    public void BackwardM2(float learningRate = .0005f)
    {
        TrainPolicyMinibatch(1, learningRate);
    }

    public void TrainPolicyMinibatch(int batchSize = 8, float learningRate = .0005f)
    {
        if (policyReplay.Count == 0) return;
        if (Weights.m2Moment == null) Weights.m2Moment = new float[32];
        int count = Mathf.Min(Mathf.Max(1, batchSize), policyReplay.Count);
        for (int b = 0; b < count; b++)
        {
            NativePolicyReplaySample sample = policyReplay[exploration.Next(policyReplay.Count)];
            float gx = Mathf.Clamp(2f * (sample.predictionX - sample.targetX) / GoalNormalizationSquared, -32f, 32f);
            float gy = Mathf.Clamp(2f * (sample.predictionY - sample.targetY) / GoalNormalizationSquared, -32f, 32f);
            float actionGradient = Mathf.Clamp(gx + gy, -4f, 4f) * .05f;
            for (int h = 0; h < Hidden; h++) { int x = h * 2; Weights.m2Moment[x] = Mathf.Clamp(.9f * Weights.m2Moment[x] + actionGradient * sample.combined[h] / count, -64f, 64f); Weights.m2Output[x] = Mathf.Clamp(Weights.m2Output[x] - learningRate * Weights.m2Moment[x], -4f, 4f); }
        }
        LastM2Loss = Mathf.Max(0f, LastM2Loss); 
    }

    // M3 is supervised only by the real physics transition.  A bounded replay
    // set makes its small online update less sensitive to the latest body.
    public bool TrainDynamics(Vector2 measuredDisplacement, float learningRate = .0005f)
    {
        if (!hasPrediction || !Weights.IsFinite() || float.IsNaN(measuredDisplacement.x) || float.IsNaN(measuredDisplacement.y) || float.IsInfinity(measuredDisplacement.x) || float.IsInfinity(measuredDisplacement.y)) { hasPrediction = false; return false; }
        dynamicsReplay.Add(new NativeDynamicsReplaySample { combined = (float[])lastM3Combined.Clone(), predictionX = lastPrediction.x, predictionY = lastPrediction.y, measuredX = measuredDisplacement.x, measuredY = measuredDisplacement.y });
        if (dynamicsReplay.Count > 128) dynamicsReplay.RemoveAt(0);
        if (Weights.m3Moment == null) Weights.m3Moment = new float[32];
        int count = Mathf.Min(8, dynamicsReplay.Count); float loss = 0f;
        for (int b = 0; b < count; b++)
        {
            NativeDynamicsReplaySample sample = dynamicsReplay[exploration.Next(dynamicsReplay.Count)];
            Vector2 error = new Vector2(sample.predictionX - sample.measuredX, sample.predictionY - sample.measuredY); loss += error.sqrMagnitude / GoalNormalizationSquared;
            float gx = Mathf.Clamp(2f * error.x / GoalNormalizationSquared, -32f, 32f), gy = Mathf.Clamp(2f * error.y / GoalNormalizationSquared, -32f, 32f);
            for (int h = 0; h < Hidden; h++) { int x = h * 2; Weights.m3Moment[x] = Mathf.Clamp(.9f * Weights.m3Moment[x] + gx * sample.combined[h] / count, -64f, 64f); Weights.m3Moment[x + 1] = Mathf.Clamp(.9f * Weights.m3Moment[x + 1] + gy * sample.combined[h] / count, -64f, 64f); Weights.m3Output[x] = Mathf.Clamp(Weights.m3Output[x] - learningRate * Weights.m3Moment[x], -4f, 4f); Weights.m3Output[x + 1] = Mathf.Clamp(Weights.m3Output[x + 1] - learningRate * Weights.m3Moment[x + 1], -4f, 4f); }
        }
        LastM3Loss = loss / count;
        hasPrediction = false;
        return true;
    }

    public void CopyM3From(NativeBrainWeights source)
    {
        Weights.CopyM3From(source);
    }
    public void RepairFrom(NativeBrainWeights source)
    {
        Weights.CopyFrom(source); Array.Clear(M1Hidden, 0, M1Hidden.Length); Array.Clear(M2Hidden, 0, M2Hidden.Length); Array.Clear(M3Hidden, 0, M3Hidden.Length); hasPrediction = false;
    }

    private float[] RunM1(float[] input)
    {
        float[] gates = Dense(input, Weights.m1Input, 28, Hidden * 3, false);
        for (int h = 0; h < Hidden; h++)
        {
            float z = Sigmoid(gates[h]); float r = Sigmoid(gates[Hidden + h]); float n = (float)Math.Tanh(gates[Hidden * 2 + h] + r * M1Hidden[h]);
            M1Hidden[h] = (1f - z) * n + z * M1Hidden[h];
        }
        return Dense(M1Hidden, Weights.m1Output, Hidden, 2, true);
    }

    public static float[] PeriodicPhase(int controlStep)
    {
        float[] phase = new float[M2PhasePeriods.Length * 2];
        for (int i = 0; i < M2PhasePeriods.Length; i++) { float angle = 2f * Mathf.PI * controlStep / M2PhasePeriods[i]; phase[i * 2] = Mathf.Sin(angle); phase[i * 2 + 1] = Mathf.Cos(angle); }
        return phase;
    }

    private static float[] BaseLimbFeatures(Limb limb)
    {
        Dictionary<string, float> input = limb.GetLocalInputs();
        return new[] { input["joint_angle"] / 180f, input["angular_velocity"] / 720f, input["touch_self"], input["touch_other_creature"], input["touch_environment"], limb.ExecutedIntervalDelta / MaximumActionDegrees, limb.dimensions.x / 3f, limb.dimensions.y / 1.2f, limb.maxMotorTorque / 200f, limb.depth / 4f, limb.MinAngle / 180f, limb.MaxAngle / 180f, limb.AtJointLimit ? 1f : 0f };
    }

    private static float[] M2LimbFeatures(Limb limb, Vector2 goal, int controlStep)
    {
        float[] baseFeatures = BaseLimbFeatures(limb); float[] phase = PeriodicPhase(controlStep); float[] result = new float[M2InputSize];
        Array.Copy(baseFeatures, result, baseFeatures.Length); Array.Copy(phase, 0, result, baseFeatures.Length, phase.Length); return result;
    }

    private static float[] Dense(float[] input, float[] weights, int inputSize, int outputSize, bool tanh)
    {
        float[] output = new float[outputSize]; for (int o = 0; o < outputSize; o++) { float value = 0f; for (int i = 0; i < inputSize && i < input.Length; i++) value += input[i] * weights[i * outputSize + o]; output[o] = tanh ? (float)Math.Tanh(value) : value; } return output;
    }
    private static float DenseWithContextScalar(float[] encoded, float[] context, float[] history, float[] output)
    {
        float value = 0f; for (int h = 0; h < Hidden; h++) value += (float)Math.Tanh(encoded[h] + context[h] + history[h]) * output[h * 2]; return (float)Math.Tanh(value);
    }
    private static float[] DenseWithContextVector(float[] encoded, float[] context, float[] history, float[] output, float action)
    {
        float[] values = new float[2]; for (int o = 0; o < 2; o++) { float value = action / MaximumActionDegrees; for (int h = 0; h < Hidden; h++) value += (float)Math.Tanh(encoded[h] + context[h] + history[h]) * output[h * 2 + o]; values[o] = (float)Math.Tanh(value); } return values;
    }
    private static float[] RunGru(float[] input, float[] state, float[] weights)
    {
        float[] next = new float[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            float z = 0f, r = 0f, n = 0f;
            for (int i = 0; i < Hidden; i++) { z += input[i] * weights[i * Hidden + h]; r += input[i] * weights[Hidden * Hidden + i * Hidden + h]; n += input[i] * weights[Hidden * Hidden * 2 + i * Hidden + h]; }
            z = Sigmoid(z); r = Sigmoid(r); next[h] = (1f - z) * (float)Math.Tanh(n + r * state[h]) + z * state[h];
        }
        Array.Copy(next, state, Hidden); return next;
    }
    private static float Sigmoid(float value) => 1f / (1f + Mathf.Exp(-Mathf.Clamp(value, -20f, 20f)));
    private static float BoundedAction(float raw)
    {
        float normalized = Mathf.Abs((float)Math.Tanh(raw));
        float magnitude = Mathf.Lerp(MinimumActionDegrees, MaximumActionDegrees, normalized);
        return raw < 0f ? -magnitude : magnitude;
    }
    private float NextGaussian() { double u1 = Math.Max(double.Epsilon, exploration.NextDouble()); double u2 = exploration.NextDouble(); return (float)(Math.Sqrt(-2d * Math.Log(u1)) * Math.Cos(2d * Math.PI * u2)); }

    public NativeCreatureSave Capture() => new NativeCreatureSave { weights = Weights, hidden = M1Hidden, m2Hidden = M2Hidden, m3Hidden = M3Hidden, lossSum = M2LossSum, lossSamples = M2LossSamples, history = frameHistory.Select(frame => new NativeHistoryFrame { values = (float[])frame.Clone() }).ToList(), historyCount = historyCount, dynamicsReplay = dynamicsReplay, policyReplay = policyReplay };
    public void Restore(NativeCreatureSave save) { if (save == null) return; Array.Copy(save.hidden ?? new float[Hidden], M1Hidden, Mathf.Min(Hidden, save.hidden?.Length ?? 0)); Array.Copy(save.m2Hidden ?? new float[Hidden], M2Hidden, Mathf.Min(Hidden, save.m2Hidden?.Length ?? 0)); Array.Copy(save.m3Hidden ?? new float[Hidden], M3Hidden, Mathf.Min(Hidden, save.m3Hidden?.Length ?? 0)); if (save.history != null) for (int i = 0; i < Mathf.Min(frameHistory.Length, save.history.Count); i++) Array.Copy(save.history[i].values ?? Array.Empty<float>(), frameHistory[i], Mathf.Min(frameHistory[i].Length, save.history[i].values?.Length ?? 0)); dynamicsReplay.Clear(); if (save.dynamicsReplay != null) dynamicsReplay.AddRange(save.dynamicsReplay.Take(128)); policyReplay.Clear(); if (save.policyReplay != null) policyReplay.AddRange(save.policyReplay.Take(128)); M2LossSum = save.lossSum; M2LossSamples = save.lossSamples; historyCount = save.historyCount; }
}

[Serializable]
public sealed class NativeCreatureSave
{
    public NativeBrainWeights weights;
    public float[] hidden;
    public float[] m2Hidden;
    public float[] m3Hidden;
    public float lossSum;
    public int lossSamples;
    public List<NativeHistoryFrame> history;
    public int historyCount;
    public List<NativeDynamicsReplaySample> dynamicsReplay;
    public List<NativePolicyReplaySample> policyReplay;
}

[Serializable]
public sealed class NativeHistoryFrame { public float[] values; }

[Serializable]
public sealed class NativeDynamicsReplaySample { public float[] combined; public float predictionX; public float predictionY; public float measuredX; public float measuredY; }

[Serializable]
public sealed class NativePolicyReplaySample { public float[] combined; public float predictionX; public float predictionY; public float targetX; public float targetY; }

[Serializable]
public sealed class NativeEcosystemCheckpoint
{
    public int schema = 1;
    public const string ConfigurationFingerprint = "m1gru16-front-m2gru16-phase8-16-32-64-d4-m3gru16-d2-replayfloats-h4-l8-p4-a8-lossnorm";
    public string configuration = ConfigurationFingerprint;
    public int tick;
    public int generation;
    public int controlStep;
    public long randomState;
    public long nextEvolutionTicks;
    public NativeBrainWeights sharedWeights;
    public List<NativeCreatureCheckpoint> creatures = new List<NativeCreatureCheckpoint>();
}

[Serializable]
public sealed class NativeCreatureCheckpoint
{
    public string creatureId;
    public string speciesId;
    public string m1SpeciesId;
    public string bodyGenomeJson;
    public NativeCreatureSave brain;
    public float bodyFitness;
    public float deathPenalty;
    public float m1EnergyFitness;
    public float previousEnergy;
    public bool deathRecorded;
}
