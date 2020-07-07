using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Spark.ML.Feature;
using Microsoft.Spark.Sql;
using Newtonsoft.Json;

namespace sparky
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("Args!");
                return;
            }
            
            var sourceDir = args[0];
            var tempDir = args[1];
            
            Console.WriteLine($"sourceDir: '{sourceDir}', tempDir: '{tempDir}'");
            
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);

            Directory.CreateDirectory(tempDir);
            
            
            var spark = SparkSession
                .Builder()
                .GetOrCreate();

            // step one train the model
            var documents = GetSourceFiles(sourceDir);
            var json = JsonConvert.SerializeObject(documents);

            var allDocumentsPath = Path.Join(tempDir, "code.json");
            File.WriteAllText(allDocumentsPath, json);

            var sourceDocuments = spark.Read().Json(allDocumentsPath);

            var tokenizer = new Tokenizer();
            var words = tokenizer
                                        .SetInputCol("Content")
                                        .SetOutputCol("words")
                                        .Transform(sourceDocuments);
            
            words.Select("words").Show(10, 1000);

            var hashingTF = new HashingTF()
                                    .SetInputCol("words")
                                    .SetOutputCol("rawFeatures")
                                    .SetNumFeatures(100000);

            var featurizedData = hashingTF.Transform(words);

            var idf = new IDF().SetInputCol("rawFeatures").SetOutputCol("features");
            var idfModel = idf.Fit(featurizedData);

            var rescaled = idfModel.Transform(featurizedData);
            var filtered = rescaled.Select("Path", "features");
            
            tokenizer.Save(Path.Join(tempDir, "temp.tokenizer"));
            hashingTF.Save(Path.Join(tempDir, "temp.hashingTF"));
            filtered.Write().Mode("overwrite").Parquet(Path.Join(tempDir, "temp.parquet"));
            idfModel.Save(Path.Join(tempDir, "temp.idfModel"));
            
            Console.WriteLine("Passing off now!");
        }

        private static List<Document> GetSourceFiles(string gitSpark)
        {
            var documents = new List<Document>();

            foreach (var file in new DirectoryInfo(gitSpark).EnumerateFiles("*.cs",
                SearchOption.AllDirectories))
                documents.Add(new Document
                {
                    Path = file.FullName,
                    Content = File.ReadAllText(file.FullName)
                });

            return documents;
        }

        private class Document
        {
            public string Content;
            public string Path;
        }
    }
}