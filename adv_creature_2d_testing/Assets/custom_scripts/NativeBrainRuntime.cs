using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

[Serializable]
public sealed class NativeBrainWeights
{
    public int version = 2;
    public float[] m1Input;
    public float[] m1Hidden;
    public float[] m1Output;
    public float[] m2Input;
    public float[] m2Middle;
    public float[] m2Deep;
    public float[] m2Extra;
    public float[] m2Ultra;
    public float[] m2Final;
    public float[] m2History;
    public float[] m2Output;
    public float[] m2Refine;
    public float[] m3Input;
    public float[] m3Middle;
    public float[] m3Deep;
    public float[] m3Extra;
    public float[] m3History;
    public float[] m3Output;
    public float[] m3State;
    public float[] m3Action;
    public float[] m2Moment;
    public float[] m3Moment;

    public static NativeBrainWeights Create(int seed)
    {
        System.Random random = new System.Random(seed);
        NativeBrainWeights w = new NativeBrainWeights {
            m1Input = Create(random, 28 * 16 * 3), m1Hidden = Create(random, 16 * 16 * 3), m1Output = Create(random, 16 * 2),
            m2Input = Create(random, 21 * 16), m2Middle = Create(random, 32 * 16), m2Deep = Create(random, 16 * 16), m2Extra = Create(random, 16 * 16), m2Ultra = Create(random, 16 * 16), m2Final = Create(random, 16 * 16), m2History = Create(random, 16 * 16 * 3), m2Output = Create(random, 16 * NativeCreatureModel.PlanningHorizon), m2Refine = Create(random, (16 + NativeCreatureModel.ConsequenceSize) * NativeCreatureModel.PlanningHorizon),
            m3Input = Create(random, 13 * 16), m3Middle = Create(random, 16 * 16), m3Deep = Create(random, 16 * 16), m3Extra = Create(random, 16 * 16), m3History = Create(random, 16 * 16 * 3), m3Output = Create(random, 16 * NativeCreatureModel.M3OutputSize), m3State = Create(random, NativeCreatureModel.SimulatedStateSize * NativeCreatureModel.M3OutputSize), m3Action = Create(random, NativeCreatureModel.M3OutputSize), m2Moment = new float[16 * NativeCreatureModel.PlanningHorizon], m3Moment = new float[16 * NativeCreatureModel.M3OutputSize]
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
    public void CopyFrom(NativeBrainWeights source) { m1Input = (float[])source.m1Input.Clone(); m1Hidden = (float[])source.m1Hidden.Clone(); m1Output = (float[])source.m1Output.Clone(); m2Input = (float[])source.m2Input.Clone(); m2Middle = (float[])source.m2Middle.Clone(); m2Deep = (float[])source.m2Deep.Clone(); m2Extra = (float[])source.m2Extra.Clone(); m2Ultra = (float[])source.m2Ultra.Clone(); m2Final = (float[])source.m2Final.Clone(); m2History = (float[])source.m2History.Clone(); m2Output = (float[])source.m2Output.Clone(); m2Refine = (float[])source.m2Refine.Clone(); m3Input = (float[])source.m3Input.Clone(); m3Middle = (float[])source.m3Middle.Clone(); m3Deep = (float[])source.m3Deep.Clone(); m3Extra = (float[])source.m3Extra.Clone(); m3History = (float[])source.m3History.Clone(); m3Output = (float[])source.m3Output.Clone(); m3State = (float[])source.m3State.Clone(); m3Action = (float[])source.m3Action.Clone(); m2Moment = source.m2Moment == null ? new float[16 * NativeCreatureModel.PlanningHorizon] : (float[])source.m2Moment.Clone(); m3Moment = source.m3Moment == null ? new float[16 * NativeCreatureModel.M3OutputSize] : (float[])source.m3Moment.Clone(); }
    public void Mutate(System.Random random, float amount = .05f)
    {
        foreach (float[] array in Arrays()) for (int i = 0; i < array.Length; i++) if (random.NextDouble() < .08) array[i] += (float)((random.NextDouble() - .5) * amount);
    }
    public void CopyM3From(NativeBrainWeights source)
    {
        Array.Copy(source.m3Input, m3Input, m3Input.Length); Array.Copy(source.m3Middle, m3Middle, m3Middle.Length); Array.Copy(source.m3Deep, m3Deep, m3Deep.Length); Array.Copy(source.m3Extra, m3Extra, m3Extra.Length);
        Array.Copy(source.m3History, m3History, m3History.Length); Array.Copy(source.m3Output, m3Output, m3Output.Length); Array.Copy(source.m3State, m3State, m3State.Length); Array.Copy(source.m3Action, m3Action, m3Action.Length);
        if (m3Moment == null) m3Moment = new float[16 * NativeCreatureModel.M3OutputSize]; Array.Copy(source.m3Moment ?? Array.Empty<float>(), m3Moment, Mathf.Min(m3Moment.Length, source.m3Moment?.Length ?? 0));
    }
    public bool IsFinite() => Arrays().All(array => array != null && array.All(value => !float.IsNaN(value) && !float.IsInfinity(value)));
    public IEnumerable<float[]> Arrays() { yield return m1Input; yield return m1Hidden; yield return m1Output; yield return m2Input; yield return m2Middle; yield return m2Deep; yield return m2Extra; yield return m2Ultra; yield return m2Final; yield return m2History; yield return m2Output; yield return m2Refine; yield return m3Input; yield return m3Middle; yield return m3Deep; yield return m3Extra; yield return m3History; yield return m3Output; yield return m3State; yield return m3Action; }
}

public sealed class NativeCreatureModel
{
    public const int History = 4, MaxLimbs = 8, MaxPathDepth = 4, Hidden = 16;
    public const int PlanningHorizon = 5, PlannerRollouts = 3, ActionTicks = 4, GoalWindowTicks = PlanningHorizon * ActionTicks;
    public const int M3OutputSize = 5, SimulatedStateSize = 5, ConsequenceSize = 6;
    public const int M2InputSize = 21;
    // These periods are measured in M2 decisions. Halving them preserves the
    // original real-time gait periods after each decision now lasts 4 ticks.
    public static readonly int[] M2PhasePeriods = { 4, 8, 16, 32 };
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
    private readonly System.Random exploration;
    private readonly List<NativeDynamicsReplaySample> dynamicsReplay = new List<NativeDynamicsReplaySample>();
    public float LastM2Loss { get; private set; } = float.PositiveInfinity;
    public float LastM3Loss { get; private set; } = float.PositiveInfinity;
    public float LastGroundPenalty { get; private set; }
    public int M2LossSamples { get; private set; }
    public float M2LossSum { get; private set; }
    public int DynamicsReplayCount => dynamicsReplay.Count;
    public float BodyFitness => M2LossSamples < MinimumM2FitnessSamples ? float.PositiveInfinity : M2LossSum / M2LossSamples;
    public Vector2 CurrentGoal { get; private set; }
    public int GoalTicksRemaining { get; private set; }
    public int LastPlannerRollouts { get; private set; }
    public float LastRefinementGain { get; private set; }
    public float LastPredictedGroundRisk { get; private set; }
    public float LastPredictedPredatorRisk { get; private set; }
    private float[] partialPlan = new float[PlanningHorizon * MaxLimbs];
    private int partialPlanLimbCount;
    private NativePredictedTransition pendingTransition;

    public NativeCreatureModel(NativeBrainWeights weights, int seed = 0) { Weights = weights ?? NativeBrainWeights.Create(Environment.TickCount); exploration = new System.Random(seed == 0 ? Environment.TickCount : seed); }

    // The real recurrent state advances only from observations.  Hypothetical
    // plan rollouts clone their state, which keeps MPC imagination from
    // contaminating the creature's physical-time memory.
    public NativeActionPlan PlanActions(float[] global, Limb[] limbs, int controlStep, bool torsoTouchingGround, float normalizedEnergy, bool predatorContact)
    {
        if (GoalTicksRemaining <= 0)
        {
            float[] m1 = RunM1(global);
            CurrentGoal = new Vector2(m1[0], m1[1]) * .25f;
            GoalTicksRemaining = GoalWindowTicks;
        }
        int limbCount = Mathf.Min(limbs.Length, MaxLimbs);
        float[][] encoded = EncodeM2(limbs, controlStep, out _, out _);
        AdvanceM3History(limbs);
        float[,] plan = new float[PlanningHorizon, MaxLimbs];
        for (int limb = 0; limb < limbCount; limb++)
        {
            float[] raw = Dense(encoded[limb], Weights.m2Output, Hidden, PlanningHorizon, true);
            for (int step = 0; step < PlanningHorizon; step++) plan[step, limb] = BoundedAction(raw[step]);
        }

        int arrivalStep = Mathf.Clamp(Mathf.CeilToInt(GoalTicksRemaining / (float)ActionTicks), 1, PlanningHorizon);
        NativePlanRollout rollout = RolloutPlan(limbs, plan, torsoTouchingGround, normalizedEnergy, predatorContact);
        float initialLoss = PlannerLoss(rollout, arrivalStep);
        for (int pass = 1; pass < PlannerRollouts; pass++)
        {
            RefinePlan(plan, encoded, rollout);
            rollout = RolloutPlan(limbs, plan, torsoTouchingGround, normalizedEnergy, predatorContact);
        }
        LastPlannerRollouts = PlannerRollouts;
        LastM2Loss = PlannerLoss(rollout, arrivalStep);
        LastRefinementGain = initialLoss - LastM2Loss;
        LastGroundPenalty = rollout.steps[0].groundRisk * 2f;
        LastPredictedGroundRisk = rollout.steps[0].groundRisk;
        LastPredictedPredatorRisk = rollout.steps[0].predatorRisk;
        M2LossSum += LastM2Loss; M2LossSamples++;
        ApplyPlannerGradient(plan, encoded, rollout, arrivalStep);
        pendingTransition = rollout.steps[0].prediction;
        partialPlanLimbCount = limbCount;
        for (int step = 0; step < PlanningHorizon; step++) for (int limb = 0; limb < MaxLimbs; limb++) partialPlan[step * MaxLimbs + limb] = plan[step, limb];
        GoalTicksRemaining = Mathf.Max(0, GoalTicksRemaining - ActionTicks);
        return new NativeActionPlan(plan, limbCount, CurrentGoal, arrivalStep);
    }

    private float[][] EncodeM2(Limb[] limbs, int controlStep, out float[] history, out float[] context)
    {
        int limbCount = Mathf.Min(limbs.Length, MaxLimbs);
        float[][] baseEncoding = new float[limbCount][]; float[] gruInput = new float[Hidden];
        for (int i = 0; i < limbCount; i++) { baseEncoding[i] = Dense(M2LimbFeatures(limbs[i], CurrentGoal, controlStep), Weights.m2Input, M2InputSize, Hidden, true); for (int h = 0; h < Hidden; h++) gruInput[h] += baseEncoding[i][h]; }
        if (limbCount > 0) for (int h = 0; h < Hidden; h++) gruInput[h] /= limbCount;
        history = RunGru(gruInput, M2Hidden, Weights.m2History);
        float[][] encoded = new float[limbCount][]; context = new float[Hidden];
        for (int i = 0; i < limbCount; i++)
        {
            float[] joined = new float[Hidden * 2]; Array.Copy(baseEncoding[i], joined, Hidden); Array.Copy(history, 0, joined, Hidden, Hidden);
            encoded[i] = Dense(Dense(Dense(Dense(Dense(joined, Weights.m2Middle, Hidden * 2, Hidden, true), Weights.m2Deep, Hidden, Hidden, true), Weights.m2Extra, Hidden, Hidden, true), Weights.m2Ultra, Hidden, Hidden, true), Weights.m2Final, Hidden, Hidden, true);
            for (int h = 0; h < Hidden; h++) context[h] += encoded[i][h];
        }
        if (limbCount > 0) for (int h = 0; h < Hidden; h++) context[h] /= limbCount;
        return encoded;
    }

    private void AdvanceM3History(Limb[] limbs)
    {
        int limbCount = Mathf.Min(limbs.Length, MaxLimbs);
        float[] context = new float[Hidden];
        for (int limb = 0; limb < limbCount; limb++)
        {
            float[] encoded = Dense(Dense(Dense(Dense(BaseLimbFeatures(limbs[limb]), Weights.m3Input, 13, Hidden, true), Weights.m3Middle, Hidden, Hidden, true), Weights.m3Deep, Hidden, Hidden, true), Weights.m3Extra, Hidden, Hidden, true);
            for (int h = 0; h < Hidden; h++) context[h] += encoded[h];
        }
        if (limbCount > 0) for (int h = 0; h < Hidden; h++) context[h] /= limbCount;
        RunGru(context, M3Hidden, Weights.m3History);
    }

    private NativePlanRollout RolloutPlan(Limb[] limbs, float[,] plan, bool torsoTouchingGround, float normalizedEnergy, bool predatorContact)
    {
        int limbCount = Mathf.Min(limbs.Length, MaxLimbs);
        float[][] encoded = new float[limbCount][]; float[] context = new float[Hidden];
        for (int i = 0; i < limbCount; i++)
        {
            encoded[i] = Dense(Dense(Dense(Dense(BaseLimbFeatures(limbs[i]), Weights.m3Input, 13, Hidden, true), Weights.m3Middle, Hidden, Hidden, true), Weights.m3Deep, Hidden, Hidden, true), Weights.m3Extra, Hidden, Hidden, true);
            for (int h = 0; h < Hidden; h++) context[h] += encoded[i][h];
        }
        if (limbCount > 0) for (int h = 0; h < Hidden; h++) context[h] /= limbCount;
        float[] simulatedHistory = (float[])M3Hidden.Clone();
        NativeSimulatedState state = new NativeSimulatedState { energy = normalizedEnergy, groundRisk = torsoTouchingGround ? 1f : 0f, predatorRisk = predatorContact ? 1f : 0f };
        NativePlanStep[] steps = new NativePlanStep[PlanningHorizon];
        for (int step = 0; step < PlanningHorizon; step++)
        {
            float[] history = RunGru(context, simulatedHistory, Weights.m3History);
            NativeSimulatedState inputState = state;
            float[] values = new float[M3OutputSize]; float[] combined = new float[Hidden];
            for (int limb = 0; limb < limbCount; limb++)
            {
                float[] output = M3Output(encoded[limb], context, history, inputState, plan[step, limb]);
                for (int outputIndex = 0; outputIndex < M3OutputSize; outputIndex++) values[outputIndex] += output[outputIndex];
                for (int h = 0; h < Hidden; h++) combined[h] += (float)Math.Tanh(encoded[limb][h] + context[h] + history[h]);
            }
            if (limbCount > 0) for (int outputIndex = 0; outputIndex < M3OutputSize; outputIndex++) values[outputIndex] /= limbCount;
            if (limbCount > 0) for (int h = 0; h < Hidden; h++) combined[h] /= limbCount;
            state.position += new Vector2(values[0], values[1]);
            state.energy = Mathf.Clamp01(state.energy + values[2]);
            state.groundRisk = values[3]; state.predatorRisk = values[4];
            float meanAction = 0f;
            for (int limb = 0; limb < limbCount; limb++) meanAction += plan[step, limb];
            if (limbCount > 0) meanAction /= limbCount;
            steps[step] = new NativePlanStep { position = state.position, energyDelta = values[2], groundRisk = values[3], predatorRisk = values[4], prediction = new NativePredictedTransition { combined = combined, state = inputState.ToArray(), action = meanAction / MaximumActionDegrees, predictionX = values[0], predictionY = values[1], predictionEnergy = values[2], predictionGround = values[3], predictionPredator = values[4] } };
        }
        return new NativePlanRollout(steps);
    }

    private float[] M3Output(float[] encoded, float[] context, float[] history, NativeSimulatedState state, float action)
    {
        float[] values = new float[M3OutputSize]; float[] stateValues = state.ToArray();
        for (int output = 0; output < M3OutputSize; output++)
        {
            float value = action / MaximumActionDegrees * Weights.m3Action[output];
            for (int h = 0; h < Hidden; h++) value += (float)Math.Tanh(encoded[h] + context[h] + history[h]) * Weights.m3Output[h * M3OutputSize + output];
            for (int s = 0; s < SimulatedStateSize; s++) value += stateValues[s] * Weights.m3State[s * M3OutputSize + output];
            values[output] = output == 3 || output == 4 ? Sigmoid(value) : (float)Math.Tanh(value);
        }
        return values;
    }

    private void RefinePlan(float[,] plan, float[][] encoded, NativePlanRollout rollout)
    {
        int limbCount = encoded.Length;
        for (int step = 0; step < PlanningHorizon; step++)
        {
            NativePlanStep consequence = rollout.steps[step];
            for (int limb = 0; limb < limbCount; limb++)
            {
                float[] features = { consequence.position.x - CurrentGoal.x, consequence.position.y - CurrentGoal.y, consequence.energyDelta, consequence.groundRisk, consequence.predatorRisk, plan[step, limb] / MaximumActionDegrees };
                float correction = 0f;
                for (int h = 0; h < Hidden; h++) correction += encoded[limb][h] * Weights.m2Refine[h * PlanningHorizon + step];
                for (int f = 0; f < ConsequenceSize; f++) correction += features[f] * Weights.m2Refine[(Hidden + f) * PlanningHorizon + step];
                plan[step, limb] = BoundedAction(Mathf.Atan(plan[step, limb] / MaximumActionDegrees) + (float)Math.Tanh(correction) * .35f);
            }
        }
    }

    private float PlannerLoss(NativePlanRollout rollout, int arrivalStep)
    {
        int q = Mathf.Clamp(arrivalStep, 1, PlanningHorizon) - 1; float loss = 0f;
        for (int step = 0; step < PlanningHorizon; step++)
        {
            NativePlanStep consequence = rollout.steps[step];
            float distance = (consequence.position - CurrentGoal).sqrMagnitude / GoalNormalizationSquared;
            if (q == PlanningHorizon - 1) loss += LatePlanWeights()[step] * distance;
            else if (step == q) loss += distance;
            else if (step < q) { float weight = (step + 1f) * (step + 1f) / Mathf.Max(1f, q * q); loss += .05f * weight * distance; }
            else loss += .25f * distance;
            loss += 10f * consequence.predatorRisk + 2f * consequence.groundRisk + .1f * Mathf.Max(0f, -consequence.energyDelta);
        }
        return loss;
    }

    public static float[] LatePlanWeights()
    {
        float[] weights = new float[PlanningHorizon]; float total = 0f;
        for (int step = 0; step < PlanningHorizon; step++) { weights[step] = (step + 1) * (step + 1); total += weights[step]; }
        for (int step = 0; step < PlanningHorizon; step++) weights[step] /= total;
        return weights;
    }

    // Truncated reverse accumulation carries each later horizon error back to
    // earlier planned actions.  It remains entirely inside the learned M3
    // rollout, so no Unity physics call or persistent recurrent state is used
    // during imagination.
    private void ApplyPlannerGradient(float[,] plan, float[][] encoded, NativePlanRollout rollout, int arrivalStep)
    {
        if (encoded.Length == 0) return;
        if (Weights.m2Moment == null || Weights.m2Moment.Length != Hidden * PlanningHorizon) Weights.m2Moment = new float[Hidden * PlanningHorizon];
        float[] horizonGradient = new float[PlanningHorizon];
        float carriedGradient = 0f;
        int deadline = Mathf.Clamp(arrivalStep, 1, PlanningHorizon) - 1;
        float[] lateWeights = LatePlanWeights();
        for (int step = PlanningHorizon - 1; step >= 0; step--)
        {
            NativePlanStep consequence = rollout.steps[step];
            float goalGradient = 2f * (consequence.position.x - CurrentGoal.x + consequence.position.y - CurrentGoal.y) / GoalNormalizationSquared;
            float safetyGradient = 10f * consequence.predatorRisk + 2f * consequence.groundRisk + .1f * Mathf.Max(0f, -consequence.energyDelta);
            float supervision = deadline == PlanningHorizon - 1 ? lateWeights[step] : (step == deadline ? 1f : step < deadline ? .05f : .25f);
            carriedGradient = Mathf.Clamp(carriedGradient + supervision * goalGradient + safetyGradient, -32f, 32f);
            horizonGradient[step] = carriedGradient * .01f;
        }
        for (int step = 0; step < PlanningHorizon; step++)
        {
            NativePlanStep consequence = rollout.steps[step];
            float gradient = horizonGradient[step];
            for (int limb = 0; limb < encoded.Length; limb++)
                for (int h = 0; h < Hidden; h++)
                {
                    int index = h * PlanningHorizon + step;
                    Weights.m2Moment[index] = Mathf.Clamp(.9f * Weights.m2Moment[index] + gradient * encoded[limb][h], -64f, 64f);
                    Weights.m2Output[index] = Mathf.Clamp(Weights.m2Output[index] - .0005f * Weights.m2Moment[index], -4f, 4f);
                    Weights.m2Refine[index] = Mathf.Clamp(Weights.m2Refine[index] - .00025f * gradient * encoded[limb][h], -4f, 4f);
                }
            float[] refineFeatures = { consequence.position.x - CurrentGoal.x, consequence.position.y - CurrentGoal.y, consequence.energyDelta, consequence.groundRisk, consequence.predatorRisk, plan[step, 0] / MaximumActionDegrees };
            for (int feature = 0; feature < ConsequenceSize; feature++)
            {
                int index = (Hidden + feature) * PlanningHorizon + step;
                Weights.m2Refine[index] = Mathf.Clamp(Weights.m2Refine[index] - .0005f * gradient * refineFeatures[feature], -4f, 4f);
            }
        }
    }

    public bool TrainDynamics(NativeTransitionTarget actual, float learningRate = .0005f)
    {
        if (pendingTransition == null || !Weights.IsFinite()) return false;
        dynamicsReplay.Add(new NativeDynamicsReplaySample { combined = pendingTransition.combined, state = pendingTransition.state, action = pendingTransition.action, predictionX = pendingTransition.predictionX, predictionY = pendingTransition.predictionY, predictionEnergy = pendingTransition.predictionEnergy, predictionGround = pendingTransition.predictionGround, predictionPredator = pendingTransition.predictionPredator, measuredX = actual.displacement.x, measuredY = actual.displacement.y, measuredEnergy = actual.energyDelta, measuredGround = actual.torsoGrounded ? 1f : 0f, measuredPredator = actual.predatorContact ? 1f : 0f });
        if (dynamicsReplay.Count > 256) dynamicsReplay.RemoveAt(0);
        if (Weights.m3Moment == null || Weights.m3Moment.Length != Hidden * M3OutputSize) Weights.m3Moment = new float[Hidden * M3OutputSize];
        int count = Mathf.Min(8, dynamicsReplay.Count); float loss = 0f;
        for (int batch = 0; batch < count; batch++)
        {
            NativeDynamicsReplaySample sample = dynamicsReplay[exploration.Next(dynamicsReplay.Count)];
            float[] predicted = { sample.predictionX, sample.predictionY, sample.predictionEnergy, sample.predictionGround, sample.predictionPredator };
            float[] measured = { sample.measuredX, sample.measuredY, sample.measuredEnergy, sample.measuredGround, sample.measuredPredator };
            for (int output = 0; output < M3OutputSize; output++)
            {
                float error = predicted[output] - measured[output]; loss += error * error;
                float gradient = Mathf.Clamp(2f * error / count, -16f, 16f);
                for (int h = 0; h < Hidden; h++)
                {
                    int index = h * M3OutputSize + output;
                    Weights.m3Moment[index] = Mathf.Clamp(.9f * Weights.m3Moment[index] + gradient * sample.combined[h], -64f, 64f);
                    Weights.m3Output[index] = Mathf.Clamp(Weights.m3Output[index] - learningRate * Weights.m3Moment[index], -4f, 4f);
                }
                for (int s = 0; s < SimulatedStateSize; s++)
                {
                    int index = s * M3OutputSize + output;
                    Weights.m3State[index] = Mathf.Clamp(Weights.m3State[index] - learningRate * gradient * sample.state[s], -4f, 4f);
                }
                Weights.m3Action[output] = Mathf.Clamp(Weights.m3Action[output] - learningRate * gradient * sample.action, -4f, 4f);
            }
        }
        LastM3Loss = loss / (count * M3OutputSize); pendingTransition = null; return true;
    }

    public void CopyM3From(NativeBrainWeights source)
    {
        Weights.CopyM3From(source);
    }
    public void RepairFrom(NativeBrainWeights source)
    {
        Weights.CopyFrom(source); Array.Clear(M1Hidden, 0, M1Hidden.Length); Array.Clear(M2Hidden, 0, M2Hidden.Length); Array.Clear(M3Hidden, 0, M3Hidden.Length); pendingTransition = null;
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
    public NativeCreatureSave Capture() => new NativeCreatureSave { weights = Weights, hidden = M1Hidden, m2Hidden = M2Hidden, m3Hidden = M3Hidden, lossSum = M2LossSum, lossSamples = M2LossSamples, goalX = CurrentGoal.x, goalY = CurrentGoal.y, goalTicksRemaining = GoalTicksRemaining, partialPlan = (float[])partialPlan.Clone(), partialPlanLimbCount = partialPlanLimbCount, history = frameHistory.Select(frame => new NativeHistoryFrame { values = (float[])frame.Clone() }).ToList(), historyCount = historyCount, dynamicsReplay = dynamicsReplay };
    public void Restore(NativeCreatureSave save) { if (save == null) return; Array.Copy(save.hidden ?? new float[Hidden], M1Hidden, Mathf.Min(Hidden, save.hidden?.Length ?? 0)); Array.Copy(save.m2Hidden ?? new float[Hidden], M2Hidden, Mathf.Min(Hidden, save.m2Hidden?.Length ?? 0)); Array.Copy(save.m3Hidden ?? new float[Hidden], M3Hidden, Mathf.Min(Hidden, save.m3Hidden?.Length ?? 0)); if (save.history != null) for (int i = 0; i < Mathf.Min(frameHistory.Length, save.history.Count); i++) Array.Copy(save.history[i].values ?? Array.Empty<float>(), frameHistory[i], Mathf.Min(frameHistory[i].Length, save.history[i].values?.Length ?? 0)); dynamicsReplay.Clear(); if (save.dynamicsReplay != null) dynamicsReplay.AddRange(save.dynamicsReplay.Take(256)); Array.Copy(save.partialPlan ?? Array.Empty<float>(), partialPlan, Mathf.Min(partialPlan.Length, save.partialPlan?.Length ?? 0)); partialPlanLimbCount = Mathf.Clamp(save.partialPlanLimbCount, 0, MaxLimbs); M2LossSum = save.lossSum; M2LossSamples = save.lossSamples; CurrentGoal = new Vector2(save.goalX, save.goalY); GoalTicksRemaining = Mathf.Clamp(save.goalTicksRemaining, 0, GoalWindowTicks); historyCount = save.historyCount; pendingTransition = null; }
}

public sealed class NativeActionPlan
{
    public readonly float[,] actions;
    public readonly int limbCount;
    public readonly Vector2 goal;
    public readonly int arrivalStep;
    public NativeActionPlan(float[,] actions, int limbCount, Vector2 goal, int arrivalStep) { this.actions = actions; this.limbCount = limbCount; this.goal = goal; this.arrivalStep = arrivalStep; }
    public float[] FirstAction()
    {
        float[] result = new float[NativeCreatureModel.MaxLimbs];
        for (int i = 0; i < result.Length; i++) result[i] = actions[0, i];
        return result;
    }
}

public struct NativeTransitionTarget
{
    public Vector2 displacement;
    public float energyDelta;
    public bool torsoGrounded;
    public bool predatorContact;
}

public sealed class NativePredictedTransition
{
    public float[] combined;
    public float[] state;
    public float action;
    public float predictionX, predictionY, predictionEnergy, predictionGround, predictionPredator;
}

public struct NativeSimulatedState
{
    public Vector2 position;
    public float energy, groundRisk, predatorRisk;
    public float[] ToArray() => new[] { position.x, position.y, energy, groundRisk, predatorRisk };
}

public struct NativePlanStep
{
    public Vector2 position;
    public float energyDelta, groundRisk, predatorRisk;
    public NativePredictedTransition prediction;
}

public sealed class NativePlanRollout
{
    public readonly NativePlanStep[] steps;
    public NativePlanRollout(NativePlanStep[] steps) { this.steps = steps; }
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
    public float goalX;
    public float goalY;
    public int goalTicksRemaining;
    public float[] partialPlan;
    public int partialPlanLimbCount;
    public List<NativeHistoryFrame> history;
    public int historyCount;
    public List<NativeDynamicsReplaySample> dynamicsReplay;
}

[Serializable]
public sealed class NativeHistoryFrame { public float[] values; }

[Serializable]
public sealed class NativeDynamicsReplaySample { public float[] combined; public float[] state; public float action; public float predictionX; public float predictionY; public float predictionEnergy; public float predictionGround; public float predictionPredator; public float measuredX; public float measuredY; public float measuredEnergy; public float measuredGround; public float measuredPredator; }

[Serializable]
public sealed class NativeEcosystemCheckpoint
{
    public int schema = 1;
    public const string ConfigurationFingerprint = "m1goal20-m2h5-k3-m2refine-m3action-motion-safety-transition4-m2d6-m3d4-phase4-8-16-32-replayv2-h4-l8-p4-a8";
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
