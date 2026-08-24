using System.Collections;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

public class LiveEcosystemPlayModeTests
{
    [UnityTest]
    public IEnumerator SampleSceneBootstrapsNativeEcosystemWithoutPython()
    {
        SceneManager.LoadScene("SampleScene", LoadSceneMode.Single);
        yield return null;
        float deadline = Time.realtimeSinceStartup + 10f;
        NativeEcosystemController native = null;
        while (Time.realtimeSinceStartup < deadline) {
            native = Object.FindAnyObjectByType<NativeEcosystemController>();
            if (native != null && native.Population == 10) break;
            yield return null;
        }
        Assert.NotNull(native);
        Assert.AreEqual(0, Object.FindObjectsByType<PythonBridge>(FindObjectsInactive.Exclude).Count(component => component.isActiveAndEnabled));
        Assert.AreEqual(1, Object.FindObjectsByType<EnvironmentSpawner>(FindObjectsInactive.Exclude).Length);
        Assert.AreEqual(10, native.Population, native.LastCheckpointError);
        Assert.NotNull(Object.FindAnyObjectByType<ObserverCamera>());
        Assert.NotNull(GameObject.Find("WorldWallLeft"));
        Assert.NotNull(GameObject.Find("WorldWallRight"));
        Food[] food = Object.FindObjectsByType<Food>(FindObjectsInactive.Include);
        Predator[] predators = Object.FindObjectsByType<Predator>(FindObjectsInactive.Include);
        Assert.AreEqual(10, food.Length);
        Assert.AreEqual(3, predators.Length);
        foreach (Food item in food) Assert.Less(Mathf.Abs(item.transform.position.y - WorldLayout.GroundTop), 1f);
    }
}
