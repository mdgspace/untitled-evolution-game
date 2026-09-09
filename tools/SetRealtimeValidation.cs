using UnityEngine;

internal class CommandScript : IRunCommand
{
    public void Execute(ExecutionResult result)
    {
        Time.timeScale = 1f;
        result.Log("Set Play Mode time scale to 1");
    }
}
