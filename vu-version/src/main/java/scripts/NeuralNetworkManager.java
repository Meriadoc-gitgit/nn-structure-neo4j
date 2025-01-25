package scripts;

import org.neo4j.procedure.*;
import org.neo4j.graphdb.*;

import java.util.*;
import java.util.stream.Collectors;

public class NeuralNetworkManager {

    @Context
    public GraphDatabaseService db_manager;

    @Procedure(name="nn.create_network", mode=Mode.WRITE)
    @Description("Create a neural network")
    public void create_network(
            @Name("tx") Transaction tx,
            @Name("network_structure") List<Long> network_structure,
            @Name("task_type") String task_type,
            @Name("hidden_activation") String hidden_activation,
            @Name("output_activation") String output_activation
    ) {
        long start_time = System.currentTimeMillis();
        System.out.println("Starting the creation of the network structure...");

        Random random = new Random();       // generate random weights

         // Create neurons for each layer
        for (int layer_id=0; layer_id<network_structure.size(); layer_id++) {
            long num_neurons = network_structure.get(layer_id);
            String layer_type = (layer_id==0) ? "input" : (layer_id==network_structure.size()-1) ? "output" : "hidden";

            for (int neuron_id=0;neuron_id<num_neurons;neuron_id++) {
                String activation_function = null;

                // Assign activation function
                if (layer_type.equals("hidden")) {
                    activation_function = hidden_activation;
                } else if (layer_type.equals("output")) {
                    if (output_activation!=null) {
                        activation_function = output_activation;
                    } else {
                        if (task_type.equals("classification")) {
                            activation_function = (num_neurons > 1) ? "softmax" : "sigmoid";
                        } else {
                            activation_function = "linear";
                        }
                    }
                }
                // Create the neuron in the database with an unique id per row
                Node neuron = tx.createNode(Label.label("Neuron"));
                neuron.setProperty("id", layer_id + "-" + neuron_id);
                neuron.setProperty("layer", layer_id);
                neuron.setProperty("type", layer_type);
                neuron.setProperty("bias", 0.0);
                neuron.setProperty("output", null);
                neuron.setProperty("m_bias", 0.0);
                neuron.setProperty("v_bias", 0.0);
                neuron.setProperty("activation_function", activation_function);
            }

            // Create connections between layers for the current row
            for (int layer_id2=0; layer_id2<network_structure.size()-1; layer_id2++) {
                long num_neurons_current = network_structure.get(layer_id2);
                long num_neurons_next = network_structure.get(layer_id2+1);

                for (int i=0; i<num_neurons_current; i++) {
                    for (int j=0; j<num_neurons_next; j++) {
                        // Generate a random weight
                        double weight = random.nextDouble() * 2 - 1; // Random value between -1 and 1

                        // Find the neurons and create a CONNECTED_TO relationship
                        Node fromNeuron = tx.findNode(Label.label("Neuron"), "id", layer_id2 + "-" + i);
                        Node toNeuron = tx.findNode(Label.label("Neuron"), "id", (layer_id2 + 1) + "-" + j);

                        if (fromNeuron != null && toNeuron != null) {
                            Relationship relationship = fromNeuron.createRelationshipTo(toNeuron, RelationshipType.withName("CONNECTED_TO"));
                            relationship.setProperty("weight", weight);
                        }
                    }
                }
                tx.commit(); // Commit the transaction
                System.out.println("Network structure created successfully.");

                long end_time = System.currentTimeMillis();
                System.out.println("Finished creating the network structure. Total time taken: " + (end_time - start_time) + " seconds.");
            }
        }
    }

    @Procedure(name="nn.create_inputs_row_node", mode = Mode.WRITE)
    @Description("Create input row nodes")
    public void create_inputs_row_node(
            @Name("tx") Transaction tx,
            @Name("network_structure") List<Long> network_structure,
            @Name("batch_size") long batch_size
    ) {
        // Step 1: Create Row nodes
        for (int id = 0; id < batch_size; id++) {
            tx.execute(
                "CREATE (n:Row {id: $id, type: 'inputsRow'})",
                Map.of("id", id)
            );
        }

        // Step 2: Create relationships between Row nodes and input neurons
        int layerIndex = 0; // Input layer
        long numNeurons = network_structure.get(0); // Number of neurons in the input layer

        for (int rowIndex = 0; rowIndex < batch_size; rowIndex++) {
            for (int neuronIndex = 0; neuronIndex < numNeurons; neuronIndex++) {
                String propertyName = "X_" + rowIndex + "_" + neuronIndex;

                tx.execute(
                        "MATCH (n1:Row {id: $from_id, type: 'inputsRow'}) " +
                            "MATCH (n2:Neuron {id: $to_id, type: 'input'}) " +
                            "CREATE (n1)-[:CONTAINS {output: $value, id: $inputFeatureId}]->(n2)",
                        Map.of(
                            "from_id", rowIndex,
                            "to_id", layerIndex + "-" + neuronIndex,
                            "inputFeatureId", rowIndex + "_" + neuronIndex,
                            "value", 0 // Default value for the relationship
                        )
                );
            }
        }
        tx.commit(); // Commit the transaction
    }

    @Procedure(name="nn.create_outputs_row_node", mode = Mode.WRITE)
    @Description("Create output row nodes")
    public void create_outputs_row_node(
            @Name("tx") Transaction tx,
            @Name("network_structure") List<Long> network_structure,
            @Name("batch_size") long batch_size
    ) {
        // Step 1: Create outputsRow nodes
        for (int index = 0; index < batch_size; index++) {
            tx.execute(
                "CREATE (n:Row {id: $id, type: 'outputsRow'})",
                Map.of("id", index)
            );
        }

        // Step 2: Create relationships from output neurons to outputsRow nodes
        int layerIndex = network_structure.size() - 1; // Output layer index
        long numNeurons = network_structure.get(layerIndex); // Number of neurons in the output layer

        for (int rowIndex = 0; rowIndex < batch_size; rowIndex++) {
            for (int neuronIndex = 0; neuronIndex < numNeurons; neuronIndex++) {
                String propertyName = "Y_" + rowIndex + "_" + neuronIndex;

                tx.execute(
                        "MATCH (n1:Neuron {id: $from_id, type: 'output'}) " +
                            "MATCH (n2:Row {id: $to_id, type: 'outputsRow'}) " +
                            "CREATE (n1)-[:CONTAINS {output: $value, id: $outputByRowId}]->(n2)",
                        Map.of(
                            "from_id", layerIndex + "-" + neuronIndex,
                            "to_id", rowIndex,
                            "outputByRowId", rowIndex + "_" + neuronIndex,
                            "value", 0 // Default output value
                        )
                );
            }
        }

        tx.commit(); // Commit the transaction
    }

    @Procedure(name="nn.forward_pass", mode = Mode.READ)
    @Description("Perform a forward pass through the neural network")
    public void forward_pass(
            @Name("tx") Transaction tx
    ) {
        tx.execute("""
            MATCH (row_for_inputs:Row {type: 'inputsRow'})-[inputsValue_R:CONTAINS]->(input:Neuron {type: 'input'})
            MATCH (input)-[r1:CONNECTED_TO]->(hidden:Neuron {type: 'hidden'})
            MATCH (hidden)-[r2:CONNECTED_TO]->(output:Neuron {type: 'output'})
            MATCH (output)-[outputsValues_R:CONTAINS]->(row_for_outputs:Row {type: 'outputsRow'})
            WITH DISTINCT row_for_inputs, inputsValue_R, input, r1, hidden, r2, output, outputsValues_R, row_for_outputs,
            
            SUM(COALESCE(inputsValue_R.output, 0) * r1.weight) AS weighted_sum
            SKIP 0 LIMIT 1000
            SET hidden.output = CASE 
                WHEN hidden.activation_function = 'relu' THEN CASE WHEN (weighted_sum + hidden.bias) > 0 THEN (weighted_sum + hidden.bias) ELSE 0 END
                WHEN hidden.activation_function = 'sigmoid' THEN 1 / (1 + EXP(-(weighted_sum + hidden.bias)))
                WHEN hidden.activation_function = 'tanh' THEN (EXP(2 * (weighted_sum + hidden.bias)) - 1) / (EXP(2 * (weighted_sum + hidden.bias)) + 1)
                ELSE weighted_sum + hidden.bias
            END
            
            WITH row_for_inputs, inputsValue_R, input, r1, hidden, r2, output, outputsValues_R, row_for_outputs,
            SUM(COALESCE(hidden.output, 0) * r2.weight) AS weighted_sum
            SET outputsValues_R.output = CASE 
                WHEN output.activation_function = 'softmax' THEN weighted_sum  // Temporary value; softmax applied later
                WHEN output.activation_function = 'sigmoid' THEN 1 / (1 + EXP(-(weighted_sum + output.bias)))
                WHEN output.activation_function = 'tanh' THEN (EXP(2 * (weighted_sum + output.bias)) - 1) / (EXP(2 * (weighted_sum + output.bias)) + 1)
                ELSE weighted_sum + output.bias
            END
            
            WITH COLLECT(output) AS output_neurons, COLLECT(outputsValues_R) AS outputsValues_Rs
            WITH output_neurons, outputsValues_Rs,
                 [n IN outputsValues_Rs | exp(COALESCE(n.output, 0))] AS exp_outputs,
                 [n IN output_neurons | n.activation_function] AS activation_functions
            WITH output_neurons, outputsValues_Rs, exp_outputs, activation_functions, 
                 REDUCE(sum = 0.0, x IN exp_outputs | sum + x) AS sum_exp_outputs
            UNWIND RANGE(0, SIZE(output_neurons) - 1) AS i
            UNWIND RANGE(0, SIZE(outputsValues_Rs) - 1) AS j
            WITH output_neurons[i] AS neuron, outputsValues_Rs[j] AS outputRow, exp_outputs[i] AS exp_output, 
                 activation_functions[i] AS activation_function, sum_exp_outputs
            WITH neuron, outputRow, 
                 CASE 
                     WHEN activation_function = 'softmax' THEN exp_output / sum_exp_outputs
                     ELSE outputRow.output
                 END AS adjusted_output
            SET outputRow.output = adjusted_output
        """);
        tx.commit();
    }

    @Procedure(name="nn.backward_pass_adam", mode = Mode.WRITE)
    @Description("Perform a backward pass using the Adam optimizer")
    public void backward_pass_adam(
            @Name("tx") Transaction tx,
            @Name("learning_rate") double learning_rate,
            @Name("beta1") double beta1,
            @Name("beta2") double beta2,
            @Name("epsilon") double epsilon,
            @Name("t") long t
    ) {
        // Step 1: Update output layer
        tx.execute("""
            MATCH (output:Neuron {type: 'output'})<-[r:CONNECTED_TO]-(prev:Neuron)
            MATCH (output)-[outputsValues_R:CONTAINS]->(row_for_outputs:Row {type: 'outputsRow'})
            WITH DISTINCT output, r, prev, outputsValues_R, row_for_outputs,
                 CASE 
                     WHEN output.activation_function = 'softmax' THEN outputsValues_R.output - outputsValues_R.expected_output
                     WHEN output.activation_function = 'sigmoid' THEN (outputsValues_R.output - outputsValues_R.expected_output) * outputsValues_R.output * (1 - outputsValues_R.output)
                     WHEN output.activation_function = 'tanh' THEN (outputsValues_R.output - outputsValues_R.expected_output) * (1 - outputsValues_R.output^2)
                     ELSE outputsValues_R.output - outputsValues_R.expected_output
                 END AS gradient,
                 $t AS t
            MATCH (prev)-[r:CONNECTED_TO]->(output)
            SET r.m = $beta1 * COALESCE(r.m, 0) + (1 - $beta1) * gradient * COALESCE(prev.output, 0)
            SET r.v = $beta2 * COALESCE(r.v, 0) + (1 - $beta2) * (gradient * COALESCE(prev.output, 0))^2
            SET r.weight = r.weight - $learning_rate * (r.m / (1 - POW($beta1, t))) / 
                           (SQRT(r.v / (1 - POW($beta2, t))) + $epsilon)
            SET output.m_bias = $beta1 * COALESCE(output.m_bias, 0) + (1 - $beta1) * gradient
            SET output.v_bias = $beta2 * COALESCE(output.v_bias, 0) + (1 - $beta2) * (gradient^2)
            SET output.bias = output.bias - $learning_rate * (output.m_bias / (1 - POW($beta1, t))) / 
                         (SQRT(output.v_bias / (1 - POW($beta2, t))) + $epsilon)
            SET output.gradient = gradient
        """, Map.of(
            "learning_rate", learning_rate,
            "beta1", beta1,
            "beta2", beta2,
            "epsilon", epsilon,
            "t", t
        ));

        // Step 2: Update hidden layers
        tx.execute("""
            MATCH (n:Neuron {type: 'hidden'})<-[:CONNECTED_TO]-(next:Neuron)
            WITH n, next, $t AS t
            MATCH (n)-[r:CONNECTED_TO]->(next)
            WITH n, SUM(next.gradient * COALESCE(r.weight, 0)) AS raw_gradient, t
            WITH n,
                 CASE 
                     WHEN n.activation_function = 'relu' THEN CASE WHEN n.output > 0 THEN raw_gradient ELSE 0 END
                     WHEN n.activation_function = 'sigmoid' THEN raw_gradient * n.output * (1 - n.output)
                     WHEN n.activation_function = 'tanh' THEN raw_gradient * (1 - n.output^2)
                     ELSE raw_gradient
                 END AS gradient, t
            MATCH (prev:Neuron)-[r_prev:CONNECTED_TO]->(n)
            SET r_prev.m = $beta1 * COALESCE(r_prev.m, 0) + (1 - $beta1) * gradient * COALESCE(prev.output, 0)
            SET r_prev.v = $beta2 * COALESCE(r_prev.v, 0) + (1 - $beta2) * (gradient * COALESCE(prev.output, 0))^2
            SET r_prev.weight = r_prev.weight - $learning_rate * (r_prev.m / (1 - POW($beta1, t))) / 
                                (SQRT(r_prev.v / (1 - POW($beta2, t))) + $epsilon)
            SET n.m_bias = $beta1 * COALESCE(n.m_bias, 0) + (1 - $beta1) * gradient
            SET n.v_bias = $beta2 * COALESCE(n.v_bias, 0) + (1 - $beta2) * (gradient^2)
            SET n.bias = n.bias - $learning_rate * (n.m_bias / (1 - POW($beta1, t))) / 
                         (SQRT(n.v_bias / (1 - POW($beta2, t))) + $epsilon)
            SET n.gradient = gradient
        """, Map.of(
            "learning_rate", learning_rate,
            "beta1", beta1,
            "beta2", beta2,
            "epsilon", epsilon,
            "t", t
        ));

        tx.commit(); // Commit transaction
    }

    @Procedure(name="nn.compute_loss", mode = Mode.WRITE)
    @Description("Compute the loss of the neural network")
    public double compute_loss(
            @Name("tx") Transaction tx,
            @Name("task_type") String task_type
    ) {
        Result result;

        if ("classification".equalsIgnoreCase(task_type)) {
            // Cross-Entropy Loss for Classification
            result = tx.execute("""
                MATCH (output:Neuron {type: 'output'})
                MATCH (output)-[outputsValues_R:CONTAINS]->(row_for_outputs:Row {type: 'outputsRow'})
                WITH outputsValues_R,
                     COALESCE(outputsValues_R.output, 0) AS predicted,
                     COALESCE(outputsValues_R.expected_output, 0) AS actual,
                     1e-10 AS epsilon
                RETURN SUM(
                    -actual * LOG(predicted + epsilon) - (1 - actual) * LOG(1 - predicted + epsilon)
                ) AS loss
            """);
        } else if ("regression".equalsIgnoreCase(task_type)) {
            // Mean Squared Error (MSE) for Regression
            result = tx.execute("""
                MATCH (output:Neuron {type: 'output'})
                MATCH (output)-[outputsValues_R:CONTAINS]->(row_for_outputs:Row {type: 'outputsRow'})
                WITH outputsValues_R,
                     COALESCE(outputsValues_R.output, 0) AS predicted,
                     COALESCE(outputsValues_R.expected_output, 0) AS actual
                RETURN AVG((predicted - actual)^2) AS loss
            """);
        } else {
            throw new IllegalArgumentException("Invalid task type. Supported types are 'classification' and 'regression'.");
        }

        // Fetch the loss value
        Map<String, Object> record = result.next();
        tx.commit();
        return record.getOrDefault("loss", 0.0) instanceof Double
                ? (Double) record.get("loss")
                : 0.0;
    }

    @Procedure(name="nn.initialize_adam_parameters", mode = Mode.WRITE)
    @Description("Initialize Adam optimizer parameters for each weight and bias")
    public void initialize_adam_parameters(
            @Name("tx") Transaction tx
    ) {
        tx.execute("""
             MATCH ()-[r:CONNECTED_TO]->()
            SET r.m = 0.0, r.v = 0.0
        """);
        tx.execute("""
            MATCH (n:Neuron)
            SET n.m_bias = 0.0, n.v_bias = 0.0
        """);
        tx.commit();
    }

    @Procedure(name="nn.constrain_weights", mode = Mode.WRITE)
    @Description("Constrain the weights to a specified range")
    public void constrain_weights(
            @Name("tx") Transaction tx
    ) {
        tx.execute("""
                     MATCH ()-[r:CONNECTED_TO]->()
                    SET r.weight = CASE 
                        WHEN r.weight > 1.0 THEN 1.0 
                        WHEN r.weight < -1.0 THEN -1.0 
                        ELSE r.weight 
                    END
                """);
        tx.commit();
    }

    @Procedure(name="nn.evaluate_model", mode = Mode.WRITE)
    @Description("Evaluate the model on the test set")
    public Map<String, Object> evaluate_model(Transaction tx) {
        Result result = tx.execute("""
            MATCH (n:Neuron {type: 'output'})
            RETURN n.id AS id, n.output AS predicted
        """);

        // Convert the results to a Map of id -> predicted
        return result.stream()
                .collect(Collectors.toMap(
                        record -> record.get("id").toString(),
                        record -> record.get("predicted")
                ));
    }


    @Procedure(name="nn.expected_output", mode = Mode.READ)
    @Description("Get the expected output values for the output neurons")
    public Map<String, Object> expected_output(Transaction tx) {
        Result result = tx.execute("""
            MATCH (n:Neuron {type: 'output'})
            RETURN n.id AS id, n.expected_output AS expected
        """);

        // Convert the results to a Map of id -> expected_output
        return result.stream()
                .collect(Collectors.toMap(
                        record -> record.get("id").toString(),
                        record -> record.get("expected")
                ));
    }



    @Procedure(name="nn.initialize_nn", mode = Mode.WRITE)
    @Description("Initialize the neural network")
    public void initialize_nn(
            @Name("network_structure") List<Long> network_structure,
            @Name("task_type") String task_type,
            @Name("activation") String activation,
            @Name("batch_size") int batch_size
    ) {
        Transaction tx = db_manager.beginTx();

        if (task_type.equals("regression")) {
            tx.execute("""
                    CALL nn.create_network($networkStructure, $task_type, null, $activation)
                """, Map.of(
                "network_structure", network_structure,
                "task_type", task_type,
                "activation", activation
            ));
        } else if (task_type.equals("classification")) {
            tx.execute("""
                    CALL nn.create_network($networkStructure, $task_type, $activation, null)
                """, Map.of(
                "networkStructure", network_structure,
                "task_type", task_type,
                "activation", activation
            ));
        }

        tx.commit();
        System.out.println(String.format("Network created: structure=%s, task_type=%s, activation=%s",
                network_structure, task_type, activation));


        long startTime = System.currentTimeMillis(); // Record the start time
        System.out.println("Starting to set batch inputs/expected rows...");

        // Create input rows
        tx.execute("""
                CALL nn.create_inputs_row_node($networkStructure, $batchSize)
            """, Map.of(
            "networkStructure", network_structure,
            "batchSize", batch_size
        ));
        tx.commit();
        System.out.println("Input rows created.");

        // Create output rows
        tx.execute("""
                CALL nn.create_outputs_row_node($networkStructure, $batchSize)
            """, Map.of(
            "networkStructure", network_structure,
            "batchSize", batch_size
        ));
        tx.commit();
        System.out.println("Output rows created.");

        long endTime = System.currentTimeMillis(); // Record the end time
        double duration = (endTime - startTime) / 1000.0; // Calculate the duration in seconds
        System.out.println(String.format("Finished setting batch inputs/expected rows. Total time taken: %.2f seconds.", duration));
        tx.close();
    }

    @Procedure(name="nn.setInputs_expectedOutputs", mode = Mode.WRITE)
    @Description("Set the input and expected output values for the neural network")
    public void setInputs_expectedOutputs(
            @Name("dataset")Map<String, Object> dataset
    ) {
        long startTime = System.currentTimeMillis(); // Record the start time
        System.out.println("Starting to set inputs/expected values and Adam parameters of the network...");

        Transaction tx = db_manager.beginTx();
        try {
            // Step 1: Initialize Adam parameters
            tx.execute("CALL nn.initialize_adam_parameters", Map.of());
            System.out.println("Adam parameters initialized.");

            // Step 2: Set input values
            tx.execute("CALL nn.set_inputs($dataset)", Map.of("dataset", dataset));
            System.out.println("Inputs set.");

            // Step 3: Set expected output values
            tx.execute("CALL nn.set_expected_outputs($dataset)", Map.of("dataset", dataset));
            System.out.println("Expected outputs set.");

            // Commit transaction
            tx.commit();

            long endTime = System.currentTimeMillis(); // Record the end time
            double duration = (endTime - startTime) / 1000.0; // Calculate the duration in seconds
            System.out.println(String.format("Finished setting inputs/expected values and Adam parameters. Total time taken: %.2f seconds.", duration));
        } catch (Exception e) {
            tx.rollback(); // Rollback transaction on error
            System.err.println("Error setting inputs/expected outputs: " + e.getMessage());
            e.printStackTrace();
        } finally {
            tx.close(); // Ensure the transaction is closed
        }
    }



    // Method to normalize inputs (dummy method to match the Python logic)
    private List<Double> normalize(List<Double> rawInputs) {
        // Add normalization logic here (e.g., min-max scaling, standardization)
        return rawInputs; // Placeholder: Return raw inputs as-is for now
    }

    @Procedure(name="nn.set_inputs", mode = Mode.WRITE)
    @Description("Set the input values for the neural network")
    // Set inputs in the database
    public void set_inputs(
            @Name("tx") Transaction tx,
            @Name("dataset") List<Map<String, Object>> dataset
    ) {
        for (int rowIndex = 0; rowIndex < dataset.size(); rowIndex++) {
            // Extract raw inputs from the dataset row
            Map<String, Object> row = dataset.get(rowIndex);
            List<Double> rawInputs = (List<Double>) row.get("inputs");

            // Normalize the inputs
            List<Double> normalizedInputs = normalize(rawInputs);

            // Iterate over the normalized inputs and update the database
            for (int i = 0; i < normalizedInputs.size(); i++) {
                double value = normalizedInputs.get(i);
                String propertyName = String.format("X_%d_%d", rowIndex, i);

                String query = """
                    MATCH (row:Row {type: 'inputsRow', id: $rowId})-[r:CONTAINS {id: $inputFeatureId}]->(inputs:Neuron {type: 'input', id: $inputNeuronId})
                    SET r.output = $value
                """;

                // Execute the query
                tx.execute(query, Map.of(
                        "rowId", String.valueOf(rowIndex),
                        "inputFeatureId", String.format("%d_%d", rowIndex, i),
                        "inputNeuronId", String.format("0-%d", i),
                        "value", value
                ));
            }
        }
    }

    @Procedure(name="nn.set_expected_outputs", mode = Mode.WRITE)
    @Description("Set the expected output values for the neural network")
    public void set_expected_outputs(
            @Name("tx") Transaction tx,
            @Name("dataset") List<Map<String, Object>> dataset,
            @Name("output_layer_index") int output_layer_index
    ) {
        for (int rowIndex = 0; rowIndex < dataset.size(); rowIndex++) {
            // Extract expected outputs from the dataset row
            Map<Integer, Double> expectedOutputs = (Map<Integer, Double>) dataset.get(rowIndex).get("expected_outputs");

            // Iterate over the expected outputs and update the database
            for (Map.Entry<Integer, Double> entry : expectedOutputs.entrySet()) {
                int i = entry.getKey(); // Neuron index
                double value = entry.getValue(); // Expected output value

                String query = """
                    MATCH (:Neuron {type: 'output', id: $outputNeuronId})-[r:CONTAINS {id: $predictedOutputId}]->(row:Row {type: 'outputsRow', id: $rowId})
                    SET r.expected_output = $value
                """;

                // Execute the query
                tx.execute(query, Map.of(
                        "rowId", String.valueOf(rowIndex),
                        "predictedOutputId", String.format("%d_%d", rowIndex, i),
                        "outputNeuronId", String.format("%d-%d", output_layer_index, i),
                        "value", value
                ));
            }
        }
    }

    @Procedure(name="nn.train", mode = Mode.WRITE)
    @Description("Train the neural network")
    public double[] train(
            @Name("tx") Transaction tx,
            @Name("dataset") List<Map<String, Object>> dataset,
            @Name("learning_rate") double learning_rate,
            @Name("beta1") double beta1,
            @Name("beta2") double beta2,
            @Name("epsilon") double epsilon,
            @Name("task_type") String task_type,
            @Name("epoch") int epoch
    ) {
        double totalTrainLoss = 0;

        try {
            // Step 1: Perform forward pass
            tx.execute("CALL nn.forward_pass");
            System.out.println("Forward pass completed.");

            // Step 2: Compute loss
            Result lossResult = tx.execute("""
                CALL nn.compute_loss($task_type)
                """, Map.of("task_type", task_type));
            double loss = (double) lossResult.next().get("loss");
            totalTrainLoss += loss;
            System.out.println(String.format("Loss for epoch %d: %.4f", epoch, loss));

            // Step 3: Perform backward pass with Adam optimization
            tx.execute("""
                CALL nn.backward_pass_adam($learning_rate, $beta1, $beta2, $epsilon, $epoch)
            """, Map.of(
                    "learning_rate", learning_rate,
                    "beta1", beta1,
                    "beta2", beta2,
                    "epsilon", epsilon,
                    "epoch", epoch
            ));
            System.out.println("Backward pass with Adam optimization completed.");

            // Step 4: Constrain weights
            tx.execute("CALL nn.constrain_weights");
            System.out.println("Weights constrained.");

            // Step 5: Calculate average loss
            int datasetSize = dataset.size(); // Derive dataset size from the list
            double avgTrainLoss = totalTrainLoss / datasetSize;
            System.out.println(String.format("Epoch %d, Train Loss: %.4f, Train AVG Loss: %.4f", epoch, loss, avgTrainLoss));

            return new double[]{loss, avgTrainLoss};
        } catch (Exception e) {
            System.err.println("Error during training: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Training failed.", e);
        }
    }

    @Procedure(name="nn.train_on_single", mode = Mode.WRITE)
    @Description("Train the neural network on a single case")
    public void train_on_single(
            @Name("case") Map<String, Object> caseData,
            @Name("epochs") int epochs,
            @Name("learning_rate") double learning_rate,
            @Name("beta1") double beta1,
            @Name("beta2") double beta2,
            @Name("epsilon") double epsilon,
            @Name("task_type") String task_type
    ) {
        // Extract inputs and expected outputs
        List<Double> rawInputs = (List<Double>) caseData.get("inputs");
        Map<Integer, Double> expectedOutputs = (Map<Integer, Double>) caseData.get("expected_outputs");

        // Normalize inputs
        List<Double> normalizedInputs = normalize(rawInputs);

        try (Transaction tx = db_manager.beginTx()) {
            // Set expected outputs
            tx.execute("""
                CALL nn.set_expected_outputs($expectedOutputs)
            """, Map.of("expectedOutputs", expectedOutputs));

            // Set inputs
            tx.execute("""
                CALL nn.set_inputs($normalizedInputs)
            """, Map.of("normalizedInputs", normalizedInputs));

            // Initialize loss tracking
            List<Double> losses = new ArrayList<>();

            // Training loop
            for (int epoch = 1; epoch <= epochs; epoch++) {
                // Forward pass
                tx.execute("CALL nn.forward_pass");

                // Compute loss
                Result result = tx.execute("""
                    CALL nn.compute_loss($task_type)
                """, Map.of("task_type", task_type));
                double loss = (double) result.next().get("loss");
                losses.add(loss);

                // Backward pass with Adam optimizer
                tx.execute("""
                    CALL nn.backward_pass_adam($learning_rate, $beta1, $beta2, $epsilon, $epoch)
                """, Map.of(
                    "learning_rate", learning_rate,
                    "beta1", beta1,
                    "beta2", beta2,
                    "epsilon", epsilon,
                    "epoch", epoch
                ));

                // Constrain weights
                tx.execute("CALL nn.constrain_weights");

                // Print progress every 100 epochs
                if ((epoch + 1) % 100 == 0) {
                    System.out.println(String.format("Epoch %d/%d, Train Loss: %.4f", epoch + 1, epochs, loss));
                }

                // Check for convergence
                if (loss < 0.01) {
                    System.out.println(String.format("Converged at epoch %d", epoch));
                    break;
                }
            }

            // Commit the transaction
            tx.commit();
        } catch (Exception e) {
            System.err.println("Error during training on a single case: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Procedure(name="nn.validate", mode = Mode.READ)
    @Description("Validate the neural network on a dataset")
    public double validate(
            @Name("dataset") List<Map<String, Object>> dataset,
            @Name("task_type") String task_type,
            @Name("epoch") int epoch
    ) {
        double totalValLoss = 0;

        try (Transaction tx = db_manager.beginTx()) {
            // Iterate through each case in the dataset
            for (int caseIndex = 0; caseIndex < dataset.size(); caseIndex++) {
                System.out.println(String.format("\n--- Validating Case %d ---", caseIndex));

                // Extract inputs and expected outputs from the current case
                Map<String, Object> singleCase = dataset.get(caseIndex);
                List<Double> rawInputs = (List<Double>) singleCase.get("inputs");
                Map<Integer, Double> expectedOutputs = (Map<Integer, Double>) singleCase.get("expected_outputs");

                // Normalize inputs
                List<Double> normalizedInputs = normalize(rawInputs);

                // Set expected outputs
                tx.execute("""
                    CALL nn.set_expected_outputs($expectedOutputs)
                """, Map.of("expectedOutputs", expectedOutputs));

                // Set inputs
                tx.execute("""
                    CALL nn.set_inputs($normalizedInputs)
                """, Map.of("normalizedInputs", normalizedInputs));

                // Perform forward pass
                tx.execute("CALL nn.forward_pass");

                // Compute loss for the current case
                Result lossResult = tx.execute("""
                    CALL nn.compute_loss($task_type)
                """, Map.of("task_type", task_type));
                double loss = (double) lossResult.next().get("loss");
                totalValLoss += loss;
            }

            // Compute average validation loss
            double avgValLoss = totalValLoss / dataset.size();
            System.out.println(String.format("Epoch %d, Validation AVG Loss: %.4f", epoch, avgValLoss));
            return avgValLoss;

        } catch (Exception e) {
            System.err.println("Error during validation: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Validation failed.", e);
        }
    }

    @Procedure(name="nn.validate_on_single", mode = Mode.READ)
    @Description("Validate the neural network on a single case")
    public void validate_on_single(
            @Name("val_data") Map<String, Object> val_data,
            @Name("epochs") int epochs,
            @Name("task_type") String task_type
    ) {
        // Extract inputs and expected outputs from validation data
        List<Double> rawInputs = (List<Double>) val_data.get("inputs");
        Map<Integer, Double> expectedOutputs = (Map<Integer, Double>) val_data.get("expected_outputs");

        // Normalize inputs
        List<Double> normalizedInputs = normalize(rawInputs);

        try (Transaction tx = db_manager.beginTx()) {
            // Set expected outputs
            tx.execute("""
                CALL nn.set_expected_outputs($expectedOutputs)
            """, Map.of("expectedOutputs", expectedOutputs));

            // Set inputs
            tx.execute("""
                CALL nn.set_inputs($normalizedInputs)
            """, Map.of("normalizedInputs", normalizedInputs));

            // Initialize loss tracking
            List<Double> losses = new ArrayList<>();

            // Validation loop
            for (int epoch = 1; epoch <= epochs; epoch++) {
                // Perform forward pass
                tx.execute("CALL nn.forward_pass");

                // Compute loss
                Result lossResult = tx.execute("""
                    CALL nn.compute_loss($task_type)
                """, Map.of("task_type", task_type));
                double loss = (double) lossResult.next().get("loss");
                losses.add(loss);

                // Print progress every 100 epochs
                if ((epoch + 1) % 100 == 0) {
                    System.out.println(String.format("Epoch %d/%d, Validation Loss: %.4f", epoch + 1, epochs, loss));
                }
            }

            // Commit transaction after all epochs
            tx.commit();
        } catch (Exception e) {
            System.err.println("Error during single validation: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Procedure(name="nn.test", mode = Mode.READ)
    @Description("Test the neural network on a dataset")
    public double test(
            @Name("test_data") List<Map<String, Object>> test_data,
            @Name("task_type") String task_type) {
        double totalTestLoss = 0;

        try (Transaction tx = db_manager.beginTx()) {
            // Iterate over each test case
            for (Map<String, Object> testCase : test_data) {
                // Extract inputs and expected outputs
                List<Double> rawInputs = (List<Double>) testCase.get("inputs");
                Map<Integer, Double> expectedOutputs = (Map<Integer, Double>) testCase.get("expected_outputs");

                // Normalize inputs
                List<Double> normalizedInputs = normalize(rawInputs);

                // Set expected outputs
                tx.execute("""
                    CALL nn.set_expected_outputs($expectedOutputs)
                """, Map.of("expectedOutputs", expectedOutputs));

                // Set inputs
                tx.execute("""
                    CALL nn.set_inputs($normalizedInputs)
                """, Map.of("normalizedInputs", normalizedInputs));

                // Perform forward pass
                tx.execute("CALL nn.forward_pass");

                // Compute loss for this case
                Result lossResult = tx.execute("""
                    CALL nn.compute_loss($task_type)
                """, Map.of("task_type", task_type));
                double loss = (double) lossResult.next().get("loss");

                // Accumulate the loss
                totalTestLoss += loss;
            }

            // Calculate average test loss
            double avgTestLoss = totalTestLoss / test_data.size();
            return avgTestLoss;

        } catch (Exception e) {
            System.err.println("Error during testing: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Testing failed.", e);
        }
    }


    // Evaluate the model
    @Procedure(name="nn.evaluate", mode = Mode.READ)
    @Description("Evaluate the neural network model")
    public void evaluate() {
        System.out.println("Evaluating model...");

        try (Transaction tx = db_manager.beginTx()) {
            // Read predictions from the database
            Result predictionsResult = tx.execute("CALL nn.evaluate_model");
            Map<String, Object> predictions = predictionsResult.next(); // Assuming single row result
            System.out.println("Predictions: " + predictions);

            // Read expected outputs from the database
            Result expectedsResult = tx.execute("CALL nn.expected_output");
            Map<String, Object> expecteds = expectedsResult.next(); // Assuming single row result
            System.out.println("Expecteds: " + expecteds);

            tx.commit();
        } catch (Exception e) {
            System.err.println("Error during evaluation: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Create batches from data
    @Procedure(name="nn.create_batches", mode = Mode.READ)
    @Description("Create batches from data")
    public List<List<Map<String, Object>>> create_batches(
            @Name("data") List<Map<String, Object>> data,
            @Name("batch_size") int batchSize,
            @Name("shuffle") boolean shuffle) {
        // Shuffle data if required
        if (shuffle) {
            Collections.shuffle(data, new Random());
        }

        // Create batches
        List<List<Map<String, Object>>> batches = new ArrayList<>();
        for (int i = 0; i < data.size(); i += batchSize) {
            int end = Math.min(i + batchSize, data.size());
            batches.add(data.subList(i, end));
        }
        return batches;
    }

    // Normalize inputs
    @Procedure(name="nn.normalized", mode = Mode.READ)
    @Description("Normalize inputs")
    public List<Double> normalized(
            @Name("input") List<Double> input,
            @Name("mms") boolean mms) {
        List<Double> normalizedInputs = new ArrayList<>();
        if (mms) {
            // Min-Max Normalization
            double min = Collections.min(input);
            double max = Collections.max(input);
            for (double value : input) {
                normalizedInputs.add((value - min) / (max - min));
            }
        } else {
            // Standardization (Z-score normalization)
            double mean = input.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double stdDev = Math.sqrt(input.stream()
                    .mapToDouble(value -> Math.pow(value - mean, 2))
                    .average()
                    .orElse(0));
            for (double value : input) {
                normalizedInputs.add((value - mean) / stdDev);
            }
        }
        return normalizedInputs;
    }
}
