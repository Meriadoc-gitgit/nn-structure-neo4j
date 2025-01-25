package scripts;

import org.neo4j.graphdb.*;
import org.neo4j.procedure.*;

import java.util.List;
import java.util.Map;

public class Neo4jDatabaseManager {

    @Context
    public GraphDatabaseService db_manager;

    // Procedure to execute a write transaction
    @Procedure(name = "db.execute", mode = Mode.WRITE)
    @Description("Executes a write transaction")
    public void execute(
            @Name("query") String query,
            @Name("parameters") Map<String, Object> parameters
    ) {
        try (Transaction tx = db_manager.beginTx()) {
            tx.execute(query, parameters);
            tx.commit();
        } catch (Exception e) {
            throw new RuntimeException("Error executing write transaction: " + e.getMessage(), e);
        }
    }

    // Procedure to execute a read transaction
    @Procedure(name = "db.execute_read", mode = Mode.READ)
    @Description("Executes a read transaction")
    public List<Map<String, Object>> execute_read(
            @Name("query") String query,
            @Name("parameters") Map<String, Object> parameters
    ) {
        try (Transaction tx = db_manager.beginTx()) {
            Result result = tx.execute(query, parameters);
            return result.stream().toList(); // Collect results as a list of maps
        } catch (Exception e) {
            throw new RuntimeException("Error executing read transaction: " + e.getMessage(), e);
        }
    }
}
