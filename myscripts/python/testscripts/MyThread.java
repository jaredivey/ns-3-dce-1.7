public class MyThread extends Thread {
    private int startIdx, nThreads, maxIdx;

    public MyThread(int s, int n, int m)
    {
        this.startIdx = s;
        this.nThreads = n;
        this.maxIdx = m;
    }

    @Override
    public void run()
        {
        for(int i = this.startIdx; i < this.maxIdx; i += this.nThreads)
        {
            System.out.println("[ID " + this.getId() + "] " + i);
        }
    }
}
