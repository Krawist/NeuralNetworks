class NeuralNetwork(val name : String, val hiddenLayerWeights : List<Array<DoubleArray>>,val hiddenLayerBiais : List<DoubleArray>, val outputLayerWeight : Array<DoubleArray>, val outputLayerBias : DoubleArray, val tauxApprentissage : Double) {

    /**
     * Retourne la trace d'exécution d'un réseau, c'est a dire les valeurs emises par les différents neuronnes de notre réseau
     */
    fun computeOnData(data : DoubleArray) : List<DoubleArray>{

        val trace : MutableList<DoubleArray> = ArrayList()

        var interneContent = data
        trace.add(data)
        hiddenLayerWeights.forEachIndexed { layerIndex, layerNeuronsWeights ->
            interneContent = forwardOperation(layerNeuronsWeights,hiddenLayerBiais[layerIndex],interneContent)
            trace.add(interneContent)
        }

        interneContent = forwardOperation(outputLayerWeight,outputLayerBias,interneContent)
        trace.add(interneContent)

        return trace

    }

    /**
     * Retourne la trace d'exécution, c'est a dire les valeurs d'erreur des différents neuronnes de notre réseau
     */
    fun backwardOnData(data: DoubleArray, previuousData : List<DoubleArray>): List<DoubleArray> {

        val trace : MutableList<DoubleArray> = ArrayList()
        trace.add(data)

        var interneContent = backwardOperation(outputLayerWeight,outputLayerBias,data,previuousData[previuousData.size-2])
        trace.add(interneContent)

        for(indexOfLayer in (1 until hiddenLayerWeights.size).reversed()){
            val layerNeuronsWeights = hiddenLayerWeights[indexOfLayer]
            val forwardData = backwardOperation(layerNeuronsWeights,hiddenLayerBiais[indexOfLayer],interneContent,previuousData[indexOfLayer])
            trace.add(forwardData)
        }

        return trace

    }

    fun updateWeight(deltaOutputLayer: DoubleArray, deltaHiddenLayer: List<DoubleArray>, forwardResult: List<DoubleArray> ) {
        // on met à jour les poids des sysnapses de la couche de sortie


        outputLayerWeight.forEachIndexed { indexOfNeurons, doubles ->
            doubles.forEachIndexed { indexOfWeigh, d ->
                outputLayerWeight[indexOfNeurons][indexOfWeigh] = d - (tauxApprentissage * forwardResult[forwardResult.size-2][indexOfNeurons] * deltaOutputLayer[indexOfNeurons])
            }
        }

        hiddenLayerWeights.forEachIndexed { indexOfLayer, arrayOfDoubleArrays ->
            arrayOfDoubleArrays.forEachIndexed { indexOfNeurons, doubles ->
                doubles.forEachIndexed { indexOfWeightToNeurons, d ->
                    doubles[indexOfWeightToNeurons] = d - (tauxApprentissage * forwardResult[indexOfLayer][indexOfWeightToNeurons] * deltaHiddenLayer[indexOfLayer][indexOfWeightToNeurons])
                }
            }
        }


/*        println("=================== NOUVEAU POIDS ========================")
        outputLayerWeight.forEachIndexed { index, doubles ->
            printArrays(doubles)
            println("\n")
        }

        hiddenLayerWeights.forEachIndexed { index, arrayOfDoubleArrays ->
            arrayOfDoubleArrays.forEach {
                printArrays(it)
                println("\n")
            }
        }*/
    }

}