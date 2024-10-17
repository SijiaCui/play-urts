/* Specific Responses for PLAP */
package ai.jni;

public class PLAPResponses {
    public int[][][][] observation;
    public double[][] reward;
    public boolean[][] done;
    // public String info;
    public int[][] resources;

    public PLAPResponses(int[][][][] observation, double reward[][], boolean done[][], int[][] resources) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        // this.info = info;
        this.resources = resources;
    }

    public void set(int[][][][] observation, double reward[][], boolean done[][], int[][] resources) {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        // this.info = info;
        this.resources = resources;
    }
}