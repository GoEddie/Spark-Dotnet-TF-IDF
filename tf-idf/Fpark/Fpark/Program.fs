// Learn more about F# at http://fsharp.org

open Microsoft.Spark.ML.Feature
open Microsoft.Spark.Sql
open Microsoft.VisualBasic
open System
open System.IO

type document = {path:string; content:string;}
type modelDetails = {model:IDFModel; features:DataFrame}

let getSourceFiles path =
    System.IO.Directory.GetFiles(path, "*.cs", EnumerationOptions(RecurseSubdirectories=true))
    |> Array.map (fun x-> {content= System.IO.File.ReadAllText x; path=x;})
    |> Newtonsoft.Json.JsonConvert.SerializeObject
    
let (+/) path1 path2 = Path.Combine(path1, path2)

let persistSparkSourceAsJson tempDir sourceDir =
    let tempPath = tempDir +/ "code.json"
    let writeToTempPath tempPath content = File.WriteAllText (tempPath, content)
   
    tempPath |> writeToTempPath <| getSourceFiles sourceDir
    tempPath

let cleanTempDirectory path =
    if Directory.Exists path then
        Directory.Delete path
    
    Directory.CreateDirectory path


[<EntryPoint>]
let main argv =
    
    let sourceDir = argv.[0]
    let tempDir = argv.[1]
        
    printfn "sourceDir: %s, tempDir: %s" sourceDir tempDir
    
    cleanTempDirectory tempDir
    
    let spark = SparkSession.Builder().GetOrCreate()
    
    let getDataFrame = spark.Read().Json(persistSparkSourceAsJson tempDir sourceDir)

    let tokenize(input) =
        let tokenizer = Tokenizer() 
        tokenizer.SetInputCol("content").SetOutputCol("words")
        tokenizer.Save(tempDir +/ "temp.tokenizer")
        tokenizer.Transform(input)
            
    let featurize(input) =
        let hashingTF = HashingTF().SetInputCol("words").SetOutputCol("rawFeatures").SetNumFeatures(10000)
        hashingTF.Save(tempDir +/ "temp.hashingTF")
        hashingTF.Transform(input)
        
    let model(input) =
        let idf = IDF().SetInputCol("rawFeatures").SetOutputCol("features")
        let idfModel = idf.Fit(input)
        idfModel.Save(tempDir +/ "temp.idfModel")
        {model=idfModel; features=input}
        
    let rescale(model)  =
        model.model.Transform(model.features)

    let simplify(dataFrame:DataFrame) =
        dataFrame.Select("path", "features")    
    
    let show(dataFrame:DataFrame) =
        dataFrame.Show()
        dataFrame
     
    let save(dataFrame:DataFrame) =
        dataFrame.Write().Mode("overwrite").Parquet <| tempDir +/ "temp.parquet"
        
    
    getDataFrame
        |> tokenize
        |> featurize
        |> model
        |> rescale
        |> simplify
        |> show
        
        
    0