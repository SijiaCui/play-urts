/* Specific Response for PLAR */
package ai.jni;

public class PLARResponse {
    public int[][][] observation;
    public double[] reward;
    public boolean[] done;
    public String info;
    public int[] resources;

    public PLARResponse(int[][][] observation, double reward[], boolean done[], String info, int[] resources) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
        this.resources = resources;
    }

    public void set(int[][][] observation, double reward[], boolean done[], String inf, int[] resources) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.info = info;
        this.resources = resources;
    }
}