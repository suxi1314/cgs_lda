/* 1, 201728016029009, Shanshan Wang */
/**
* @file KCore.cc
* @author  Shanshan Wang
* @version 0.2
* This is a program which computes sub graph of vertexes with more than K degrees.
* command line:
*   start-graphlite example/KCore.so ${GRAPHLITE_HOME}/part2-input/KCore-graph0_4w ${GRAPHLITE_HOME}/Output/out 6
*   start-graphlite example/KCore.so ${GRAPHLITE_HOME}/part2-input/KCore-graph1_4w ${GRAPHLITE_HOME}/Output/out 7
*/

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "GraphLite.h"

/** change VERTEX_CLASS_NAME(name) definition to use a different class name */
#define VERTEX_CLASS_NAME(name) KCore##name

/** the condition of stop the program */
#define EPS 0

/** VERTEX_CLASS_NAME(InputFormatter) can be kept as is */

/** The least number of degree*/
int K;

/** structure of vertex value*/
typedef struct KCoreValue{
    bool is_deleted; //if a vertex is deleted
    int current_degree; //current degree of a vertex
}KValue;

class VERTEX_CLASS_NAME(InputFormatter): public InputFormatter {
public:
    int64_t getVertexNum() {
        unsigned long long n;
        sscanf(m_ptotal_vertex_line, "%lld", &n);
        m_total_vertex= n;
        return m_total_vertex;
    }
    int64_t getEdgeNum() {
        unsigned long long n;
        sscanf(m_ptotal_edge_line, "%lld", &n);
        m_total_edge= n;
        return m_total_edge;
    }
    int getVertexValueSize() {
        m_n_value_size = sizeof(KValue);
        return m_n_value_size;
    }
    int getEdgeValueSize() {
        m_e_value_size = sizeof(double);
        return m_e_value_size;
    }
    int getMessageValueSize() {
        m_m_value_size = sizeof(bool);
        return m_m_value_size;
    }
    void loadGraph() {
        unsigned long long last_vertex;
        unsigned long long from;
        unsigned long long to;
        double weight = 0;
        
        KValue value = {false, 0};
        int outdegree = 0;
        
        const char *line= getEdgeLine();

        // Note: modify this if an edge weight is to be read
        //       modify the 'weight' variable

        sscanf(line, "%lld %lld", &from, &to);
        addEdge(from, to, &weight);

        last_vertex = from;
        ++outdegree;
        for (int64_t i = 1; i < m_total_edge; ++i) {
            line= getEdgeLine();

            // Note: modify this if an edge weight is to be read
            //       modify the 'weight' variable

            sscanf(line, "%lld %lld", &from, &to);
            if (last_vertex != from) {
                addVertex(last_vertex, &value, outdegree);
                last_vertex = from;
                outdegree = 1;
            } else {
                ++outdegree;
            }
            addEdge(from, to, &weight);
        }
        addVertex(last_vertex, &value, outdegree);
    }
};
/** VERTEX_CLASS_NAME(OutputFormatter): this is where the output is generated */
class VERTEX_CLASS_NAME(OutputFormatter): public OutputFormatter {
public:
    void writeResult() {
        int64_t vid;
        KValue value;
        char s[1024];

        for (ResultIterator r_iter; ! r_iter.done(); r_iter.next() ) {
            r_iter.getIdValue(vid, &value);
            // output undeleted vertex as sub graph
            if(!value.is_deleted){
                int n = sprintf(s, "%lld\n", (unsigned long long)vid);
                writeNextResLine(s, n);
            }
           
        }
    }
};

/** VERTEX_CLASS_NAME(Aggregator): you can implement other types of aggregation */
// An aggregator that records a double value to m compute sum
class VERTEX_CLASS_NAME(Aggregator): public Aggregator<int> {
public:
    void init() {
        m_global = 0;
        m_local = 0;
    }
    void* getGlobal() {
        return &m_global;
    }
    void setGlobal(const void* p) {
        m_global = * (int *)p;
    }
    void* getLocal() {
        return &m_local;
    }
    void merge(const void* p) {
        m_global += * (int *)p;
    }
    void accumulate(const void* p) {
        m_local += * (int *)p;
    }
};
/** every getSuperstep condition should be defined explicit*/
/** val is a local variable, it can only use in one superstep, it is initialized by the superstep 0*/
/** VERTEX_CLASS_NAME(): the main vertex program with compute() */
class VERTEX_CLASS_NAME(): public Vertex <KValue, double, bool> {
public:
    void compute(MessageIterator* pmsgs) {
        KValue val;
        //superstep0: initialize value of vertex
        if (getSuperstep() == 0) {
           val.is_deleted = false;
           // current degree is the number of all neighbors 
           val.current_degree = (int)getOutEdgeIterator().size();
        }else{
            //if all degree of vertex in neighbor supersteps unchanged, stop
            if (getSuperstep() >= 3) {
                int global_val = * (int *)getAggrGlobal(0);
                if (global_val == EPS) {
                    voteToHalt(); return;
                }
            }
            // get value in previous superstep    
            val.is_deleted = getValue().is_deleted;
            val.current_degree = getValue().current_degree;
            // if vetex is not deleted
            if(!getValue().is_deleted){
                // compute the numer of deleted neibors in previous superstep
                int sum = 0;
                while(! pmsgs->done()){
                    pmsgs->getValue();
                    sum += 1;
                    pmsgs->next();
                }
                // compute the number of current neighbors as current degree
                val.current_degree = val.current_degree - sum;
                // if current degree less than K, delete the vertex and send message to all neighbors
                if(val.current_degree < K){
                    val.is_deleted = true;
                    // transform to bool is neccessary
                    sendMessageToAllNeighbors(val.is_deleted);
                }
            }
            // compute if degree of the vertex is defferent between two supersteps
            int acc = fabs(getValue().current_degree - val.current_degree);
            // accumulate a value to m_local
            accumulateAggr(0, &acc);
        }
        * mutableValue() = val;
    }
};

/** VERTEX_CLASS_NAME(Graph): set the running configuration here */
class VERTEX_CLASS_NAME(Graph): public Graph {
public:
    VERTEX_CLASS_NAME(Aggregator)* aggregator;

public:
    // argv[0]: KCore.so
    // argv[1]: <input path>
    // argv[2]: <output path>
    // argv[3]: <K>
    void init(int argc, char* argv[]) {

        setNumHosts(5);
        setHost(0, "localhost", 1411);
        setHost(1, "localhost", 1421);
        setHost(2, "localhost", 1431);
        setHost(3, "localhost", 1441);
        setHost(4, "localhost", 1451);

        if (argc < 4) {
           printf ("Usage: %s <input path> <output path> <K>\n", argv[0]);
           exit(1);
        }

        m_pin_path = argv[1];
        m_pout_path = argv[2];
        int length = strlen(argv[3]);
        for(int i = 0; i < length; i++ ){
            if(isdigit(argv[3][i])==0){
                printf("<K> is an Integer!\n");
                exit(1);
            }
        }
        K = atoi(argv[3]);

        aggregator = new VERTEX_CLASS_NAME(Aggregator)[1];
        regNumAggr(1);
        regAggr(0, &aggregator[0]);
    }

    void term() {
        delete[] aggregator;
    }
};

/* STOP: do not change the code below. */
extern "C" Graph* create_graph() {
    Graph* pgraph = new VERTEX_CLASS_NAME(Graph);

    pgraph->m_pin_formatter = new VERTEX_CLASS_NAME(InputFormatter);
    pgraph->m_pout_formatter = new VERTEX_CLASS_NAME(OutputFormatter);
    pgraph->m_pver_base = new VERTEX_CLASS_NAME();

    return pgraph;
}

extern "C" void destroy_graph(Graph* pobject) {
    delete ( VERTEX_CLASS_NAME()* )(pobject->m_pver_base);
    delete ( VERTEX_CLASS_NAME(OutputFormatter)* )(pobject->m_pout_formatter);
    delete ( VERTEX_CLASS_NAME(InputFormatter)* )(pobject->m_pin_formatter);
    delete ( VERTEX_CLASS_NAME(Graph)* )pobject;
}
