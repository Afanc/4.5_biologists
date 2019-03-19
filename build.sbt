name := "pattern_recognition"

version := "0.1"

scalaVersion := "2.12.8"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2",
//  "org.scalanlp" %% "nak" % "1.2.1"
)
// libraryDependencies  += "be.botkop" %% "scorch" % "0.1.0"

// libraryDependencies += "com.github.transcendent-ai-labs.DynaML" % "dynaml-core_2.11" % "v1.5.3"

libraryDependencies  += "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.5"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
resolvers += "Artima Maven Repository" at "http://repo.artima.com/releases"
resolvers += "jitpack" at "https://jitpack.io"

scalacOptions in ThisBuild ++= Seq ("-JXss512m","-JXmx2G")