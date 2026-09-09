using UnityEngine;
using UnityEditor.TestTools.TestRunner.Api;

internal class CommandScript : IRunCommand
{
    sealed class Callbacks : ICallbacks
    {
        public void RunStarted(ITestAdaptor testsToRun) { }
        public void RunFinished(ITestResultAdaptor result) { Debug.Log("CODEX_EDITMODE_RESULT state=" + result.TestStatus + " pass=" + result.PassCount + " fail=" + result.FailCount + " skip=" + result.SkipCount); }
        public void TestStarted(ITestAdaptor test) { }
        public void TestFinished(ITestResultAdaptor result) { if (result.FailCount > 0) Debug.LogError("CODEX_EDITMODE_FAILURE " + result.FullName + " " + result.Message); }
    }

    public void Execute(ExecutionResult result)
    {
        var api = ScriptableObject.CreateInstance<TestRunnerApi>();
        api.RegisterCallbacks(new Callbacks());
        api.Execute(new ExecutionSettings { filters = new[] { new Filter { testMode = TestMode.EditMode, assemblyNames = new[] { "LiveEcosystem.EditModeTests" } } } });
        result.Log("Started LiveEcosystem EditMode tests");
    }
}
