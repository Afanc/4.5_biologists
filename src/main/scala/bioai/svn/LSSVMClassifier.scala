package bioai.svn

import breeze.linalg.DenseVector

import scala.collection.mutable.ListBuffer


class MnistSampleD(val digit: Int, val image: DenseVector[Double]) {}

object SVMClassifier extends App {

  val trainFilename = if (args.length > 2) args(2) else "./data/train.csv"

  val trainingSet = loadFromFile(trainFilename)


  def loadFromFile(filepath: String): List[MnistSampleD] = {
    val file = io.Source.fromFile(filepath)
    val listBuffer = ListBuffer[MnistSampleD]()
    for (line <- file.getLines) {
      val row = line.split(",").map(_.toDouble)
      listBuffer += new MnistSampleD(row.head.toInt, DenseVector(row.tail))
    }
    file.close
    listBuffer.toList
  }

}
