using System.Linq;
using UnityEngine;

internal class CommandScript : IRunCommand
{
    public void Execute(ExecutionResult result)
    {
        var creatures = Object.FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        foreach (var creature in creatures.OrderBy(c => c.creatureId))
        {
            var body = creature.torso == null ? null : creature.torso.Rigidbody;
            var brain = creature.GetComponent<CreatureBrain>();
            result.Log((creature.creatureId ?? "unknown") + " speed=" + (body == null ? "missing" : body.linearVelocity.magnitude.ToString("F3")) + " vx=" + (body == null ? "missing" : body.linearVelocity.x.ToString("F3")) + " y=" + (body == null ? "missing" : body.position.y.ToString("F2")) + " joint=" + (brain == null ? "missing" : brain.MeanJointSpeed.ToString("F2")));
        }
    }
}
