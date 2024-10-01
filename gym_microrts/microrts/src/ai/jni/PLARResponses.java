/* Specific Responses for PLAR */
package ai.jni;

public class PLARResponses {
    public int[][][][] observation;
    public double[][] reward;
    public boolean[][] done;
    // public String info;
    public int[][] resources;

    public PLARResponses(int[][][][] observation, double reward[][], boolean done[][], int[][] resources) {
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