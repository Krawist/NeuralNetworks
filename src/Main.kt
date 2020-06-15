import com.google.gson.GsonBuilder
import java.io.FileNotFoundException
import java.io.FileReader
import java.io.FileWriter
import java.io.IOException
import kotlin.random.Random

/** Données d'apprentissage **/
val inputs : Array<DoubleArray> = arrayOf(
    doubleArrayOf(0.0, 0.0, 0.0, 1.0),
    doubleArrayOf(0.0, 0.0, 1.0, 1.0),
    doubleArrayOf(0.0, 1.0, 1.0, 1.0),
    doubleArrayOf(1.0, 0.0, 1.0, 0.0)
)

/*
val inputs : Array<DoubleArray> = arrayOf(
    doubleArrayOf(2.toDouble(), (-1).toDouble())
)
*/


val expectedOutputs : Array<DoubleArray> = arrayOf(
    doubleArrayOf(0.0),
    doubleArrayOf(1.0),
    doubleArrayOf(0.0),
    doubleArrayOf(1.0)
)

/*val expectedOutputs : Array<DoubleArray> = arrayOf(
    doubleArrayOf(1.toDouble()))*/


/** données de test**/
val inputsTest : Array<DoubleArray> = arrayOf(
    doubleArrayOf(1.0, 0.0, 1.0, 1.0),
    doubleArrayOf(0.0, 1.0, 1.0, 1.0),
    doubleArrayOf(0.0, 0.0, 1.0, 1.0),
    doubleArrayOf(0.0, 1.0, 1.0, 1.0)
)



/** on initialise les poids avec des valeurs entre -1 et 1 **/

/**Réprésente les différents poids des différents neurones  audifférentes couches
 * liste(i): on a le arrays du doubleArrays des poids de la couche i
 * liste(i)(j) : on a le doubleArrays des poids du neurone j de la couche i
 */
val random = Random(1)
/*val hiddenLayerWeight : List<Array<DoubleArray>> = listOf<Array<DoubleArray>>(
    arrayOf(
        doubleArrayOf(0.2, 0.5, -0.8, 0.3),
        doubleArrayOf(0.1, -0.2, -0.4, 0.5),
        doubleArrayOf(0.2, -0.5, -0.3, 0.7),
        doubleArrayOf(-0.1, 0.5, -0.7, 0.6)
    )
)*/

val hiddenLayerWeight : List<Array<DoubleArray>> = listOf<Array<DoubleArray>>(
    arrayOf(
        doubleArrayOf(random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble()),
        doubleArrayOf(random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble()),
        doubleArrayOf(random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble()),
        doubleArrayOf(random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble())
    )
)


/*val hiddenLayerWeight : List<Array<DoubleArray>> = listOf<Array<DoubleArray>>(
    arrayOf(
        doubleArrayOf(0.5,1.5),
        doubleArrayOf((-1).toDouble(), (-2).toDouble())
    ),
    arrayOf(
        doubleArrayOf(1.toDouble(),3.toDouble()),
        doubleArrayOf((-1).toDouble(),(-4).toDouble())
    )
)*/

/**Réprésente les différents billet des différents neurones  aux différentes couches
 * liste(i): on a le doubleArrays des billets de la couche i
 * liste(i)(j) : on a le billet du neurone j de la couche i
 */
val hiddenLayerBias : List<DoubleArray> = listOf<DoubleArray>(
    doubleArrayOf(0.0, 0.0, 0.0, 0.0)
)


/** Représente les poids des sysnapses entraintes de la couche dernieres couches du réseau
 * array(i) designe le doubleArray des poids du neuronne i
 */
val outputsLayerWeight : Array<DoubleArray> = arrayOf(
    doubleArrayOf(random.nextDouble(), random.nextDouble(), random.nextDouble(), random.nextDouble())
)

/*val outputsLayerWeight : Array<DoubleArray> = arrayOf(
    doubleArrayOf(1.0, -3.0)
)*/

val outputsLayerBias : DoubleArray = doubleArrayOf(0.0)

/**##############################################
 * #        Phase d'entrainement
 *##############################################*/
const val nbTrainingIteration = 100000

private val gson = GsonBuilder().setPrettyPrinting().create()


fun main(args: Array<String>) {

    var neuralNetwork = NeuralNetwork("NetworkForNombreUnPair", hiddenLayerWeight,hiddenLayerBias,outputsLayerWeight,outputsLayerBias,0.1)

    try {
        neuralNetwork = gson.fromJson(FileReader(neuralNetwork.name + ".json"), NeuralNetwork::class.java)
        if (neuralNetwork != null) {
            trainModel(neuralNetwork)
            process(neuralNetwork)
        }
    } catch (e: FileNotFoundException) {
        trainModel(neuralNetwork)
        process(neuralNetwork)
    }

}

fun process(neuralNetwork: NeuralNetwork) {

    inputsTest.forEachIndexed { index, doubles ->
        val result = neuralNetwork.computeOnData(doubles)
        print("Le resultat pour l'entrée $index est de " )
        printArrays(result[result.size-1])
        println("\n")
    }

    val fileLocation = neuralNetwork.name + ".json"

    try {
        val fileWriter = FileWriter(fileLocation, false)
        fileWriter.write(gson.toJson(neuralNetwork))
        fileWriter.close()
    } catch (e: IOException) {
        e.printStackTrace()
    }

}

private fun trainModel(neuralNetwork: NeuralNetwork) {
    for (i in 0 until nbTrainingIteration) {
        // on itère sur les données d'entrainement, puis on calcule notre fonction cout
        //val resultOfDataSet : MutableList<List<DoubleArray>> = ArrayList()

/*        println("HiddenLayerWeight")
        neuralNetwork.hiddenLayerWeights[0].forEach {
            printArrays(it)
        }
        println("-------------------------------------------------------------------------")

        println("OutputLayerWeight")
        printArrays(neuralNetwork.outputLayerWeight[0])
        println("-------------------------------------------------------------------------")*/

        inputs.forEachIndexed { index, data ->

            // on parcours le réseau avec notre entrée
            val forwardResult = neuralNetwork.computeOnData(data)

/*            println("Entrée")
            printArrays(data)
            println("-------------------------------------------------------------------------")

            println("Sortie")
            printArrays(forwardResult[forwardResult.size-1])
            println("-------------------------------------------------------------------------")*/

            //resultOfDataSet.add(result)

            //println("resultat \n")
            //printArrays(result[result.size - 1])

            //printArrays(result[result.size-1])

            // on calcule l'erreur de chaque neurone de la couche de sortie
            val deltaOutputLayer = getCoutOfData(index, forwardResult[forwardResult.size - 1])
            //print("Erreur data ${index + 1}")
            //printArrays(deltaOutputLayer)
            //println("\n")

            //on fabrique le tableau des valeurs précédente
            // on itère pour avoir le cout d'erreur des couches intermediaires
            val previousDataMatrix: MutableList<DoubleArray> = ArrayList()
            for (i in (0 until forwardResult.size - 1)) {
                previousDataMatrix.add(forwardResult[i])
            }

            // on calcule l'erreur de toutes les couches intermediaire.
            val deltaHiddenLayer = neuralNetwork.backwardOnData(deltaOutputLayer, forwardResult)
            // lors de la mise a jour des poids je pars de la gauche vers la droite, hors la liste des deltaHiddenLayer en encore à l'envers, il faut la renverser
            val newDeltaHiddenLayout = deltaHiddenLayer.reversed()

            //on met à jour les différents poids de nos synapses
            neuralNetwork.updateWeight(deltaOutputLayer, newDeltaHiddenLayout, forwardResult)

        }

        //println("\n")
        //println("-----------------------------------------------------------------------------------------------")

    }
}


fun getCoutOfData(indexOfTrainingData: Int, result: DoubleArray): DoubleArray {

    // on reverra le cout pour chaque neuronne de notre sortie
    val cout = DoubleArray(result.size)

    expectedOutputs[indexOfTrainingData].forEachIndexed { indexOfNeurons, expectedValueOfTheNeurons ->
        //cout[indexOfNeurons] = 0.5 * Math.pow(expectedValueOfTheNeurons - result[indexOfNeurons],2.0)
        //cout[indexOfNeurons] = (expectedValueOfTheNeurons - result[indexOfNeurons]) * sigmoidPrime(sigmoidPrime(result[indexOfNeurons]))

        if((result[indexOfNeurons] - expectedValueOfTheNeurons)>=0){
            cout[indexOfNeurons] = 0.5 * Math.pow((result[indexOfNeurons] - expectedValueOfTheNeurons),2.0)
        }else{
            cout[indexOfNeurons] = -0.5 * Math.pow((result[indexOfNeurons] - expectedValueOfTheNeurons),2.0)
        }
    }

    return cout
}

fun printArrays(result: DoubleArray) {
    print("[")
    result.forEach {
        print(" $it ")
    }
    print("]")
    println("\n")
}



fun forwardOperation(weightMatrix : Array<DoubleArray>, biaisMatrix : DoubleArray, dataMatrix: DoubleArray): DoubleArray {

    //val trace : MutableList<DoubleArray> = ArrayList()

    val result = DoubleArray(weightMatrix.size)
    weightMatrix.forEachIndexed { neuronsIndex, weightArrays ->
        var resultInter : Double = 0.toDouble()
        dataMatrix.forEachIndexed { weightIndex, dataValue ->
            resultInter += weightArrays[weightIndex] * dataValue
        }
        result[neuronsIndex] = sigmoid(resultInter + biaisMatrix[neuronsIndex])
        //result[neuronsIndex] = (resultInter + biaisMatrix[neuronsIndex])
    }
    //println("Tableau apres le hiddenLayer ${printArrays(result)}")

    return result
}


fun backwardOperation(weightMatrix : Array<DoubleArray>, biaisMatrix : DoubleArray, newDataMatrix: DoubleArray, previousDataMatrix : DoubleArray): DoubleArray {

    val newWeightMatrix = transposeMatrix(weightMatrix)

    val result = DoubleArray(newWeightMatrix.size)


    newWeightMatrix.forEachIndexed { neuronsIndex, weightArrays ->
        var resultInter : Double = 0.toDouble()

        newDataMatrix.forEachIndexed { weightIndex, dataValue ->
            resultInter += weightArrays[weightIndex] * dataValue
        }

         result[neuronsIndex] = (sigmoidPrime(previousDataMatrix[neuronsIndex])) * resultInter

        //result[neuronsIndex] = (resultInter + biaisMatrix[neuronsIndex])
    }
    //println("Tableau apres le hiddenLayer ${printArrays(result)}")

    return result
}

fun transposeMatrix(matrix: Array<DoubleArray>): Array<DoubleArray> {

    val newMatrix = Array(matrix[0].size){DoubleArray(matrix.size)}

    matrix.forEachIndexed{indexLine: Int, doubles: DoubleArray ->
        doubles.forEachIndexed { indexColumns, d ->
            newMatrix[indexColumns][indexLine] = d
        }
    }

    return newMatrix
}


fun sigmoid(x : Double) : Double{
    return 1/(1+Math.exp(-x))
    //return x * x
}

fun sigmoidPrime(x : Double) : Double{
    return x * (1-x)
    //return 2 * x
}

