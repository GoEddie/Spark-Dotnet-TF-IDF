using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Microsoft.Spark.ML.Feature;
using Microsoft.Spark.Sql;
using Microsoft.Spark.Sql.Types;
using static Microsoft.Spark.Sql.Functions;

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
            var searchTerm = args[1];
            
            var spark = SparkSession
                .Builder()
                .GetOrCreate();

            // step one train the model

            var hashingTF = new HashingTF()
                .SetInputCol("words")
                .SetOutputCol("rawFeatures")
                .SetNumFeatures(100000);

            var tokenizer = new Tokenizer()
                .SetInputCol("Content")
                .SetOutputCol("words");
            
            var idf = new IDF()
                .SetInputCol("rawFeatures")
                .SetOutputCol("features");

                        
            var sourceDocuments = GetSourceFiles(sourceDir).toDF();
            sourceDocuments.Show();
            
            var words = tokenizer.Transform(sourceDocuments);
            var featurizedData = hashingTF.Transform(words);
            
            var idfModel = idf.Fit(featurizedData);
            
            var rescaled = idfModel.Transform(featurizedData);
            var filtered = rescaled.Select("Path", "features");
            
            var normalized = filtered.WithColumn("norm", udfCalcNorm(Col("features")));

            var searchTermDataFrame = spark.CreateDataFrame(new List<string>() {searchTerm}).WithColumnRenamed("_1", "Content");
            
            searchTermDataFrame.Show();
            var searchWords = tokenizer.Transform(searchTermDataFrame);

            var featurizedSeachTerm = hashingTF.Transform(searchWords);
            
            var search = idfModel.Transform(featurizedSeachTerm).WithColumnRenamed("features", "features2").WithColumn("norm2", udfCalcNorm(Col("features2")));
            search.Show();
            
            var results = search.CrossJoin(normalized);
            results
                .WithColumn("similarity",udfCosineSimilarity(Col("features"), Col("features2"), Col("norm"), Col("norm2")))
                .Select("path", "similarity")
                .Filter("similarity > 0.0")
                .OrderBy(Desc("similarity"))
                .Limit(10)
                .WithColumn("Search Term", Lit(searchTerm))
                .Show(10, 100000);
        }
        
        private static readonly Func<Column, Column> udfCalcNorm = Udf<Row, double>(row =>
            {
                var values = (ArrayList) row.Values[3];
                var norm = 0.0;
    
                foreach (var value in values)
                {
                    var d = (double) value;
                    norm += d * d;
                }

                return Math.Sqrt(norm);
            }
        );

        private static readonly Func<Column, Column, Column, Column, Column> udfCosineSimilarity =
            Udf<Row, Row, double, double, double>(
                (vectorA, vectorB, normA, normB) =>
                {
                    var indicesA = (ArrayList) vectorA.Values[2];
                    var valuesA = (ArrayList) vectorA.Values[3];

                    var indicesB = (ArrayList) vectorB.Values[2];
                    var valuesB = (ArrayList) vectorB.Values[3];

                    var dotProduct = 0.0;
            
                    for (var i = 0; i < indicesA.Count; i++)
                    {
                        var valA = (double) valuesA[i];

                        var indexB = findIndex(indicesB, 0, (int) indicesA[i]);

                        double valB = 0;
                        if (indexB != -1)
                        {
                            valB = (double) valuesB[indexB];
                        }
                        else
                        {
                            valB = 0;
                        }

                        dotProduct += valA * valB;
                    }

                    var divisor = normA * normB;
            
                    return divisor == 0 ? 0 : dotProduct / divisor;
                });

        private static int findIndex(ArrayList list, int currentIndex, int wantedValue)
        {
            for (var i = currentIndex; i < list.Count; i++)
                if ((int) list[i] == wantedValue)
                    return i;

            return -1;
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
    }
    
    internal class Document
    {
        public string Content;
        public string Path;
    }
    
    internal static class ToDFExtensions{
        public static DataFrame toDF(this List<Document> docs){

            var rows = new List<GenericRow>();

            var spark = SparkSession.Active();
            
            foreach(var row in docs){
                rows.Add(new GenericRow(new object[]{row.Path, row.Content}));
            }
                
            var schema = new StructType(new List<StructField>()
            {
                new StructField("Path", new StringType()),
                new StructField("Content", new StringType())
            });
            
            return spark.CreateDataFrame(rows, schema);
        }
        
    }
}