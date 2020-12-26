package main

import (
	"NN"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"time"
)

func MNIST(NN1 NN.Nodes) *NN.Nodes {
	var in, out [][]float64
	count := 0
	in = NN.MakeSimple(3, 784)
	train, err := os.Open("BigTrainSample.csv")
	if err != nil {
		log.Fatalln("couldnt open file", err)
	}
	defer train.Close()
	reader := csv.NewReader(train)
	for {
		train, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		for i := 1; i < 783; i++ {
			val, errStr := strconv.ParseFloat(train[i], 64)
			if errStr != nil {
				log.Fatalln("casting error", errStr)
			}
			in[i-1][0] = (val / 255) * 0.9
		}
		val, errStr := strconv.Atoi(train[0])
		if errStr != nil {
			log.Fatalln("casting error", errStr)
		}
		out = NN.MakeSimple(val, 10)
		NN1 = *NN.Train(in, out, 0.00000004, NN1)
		if count%1000 == 0 {
			fmt.Println(count)
			fmt.Println(NN.Query(in, NN1))
		}
		count++
	}

	test, err2 := os.Open("BigTestSample.csv")
	if err != nil {
		log.Fatalln("couldnt open file", err2)
	}
	defer test.Close()
	reader2 := csv.NewReader(test)

	var correct, total = 0, 0
	fmt.Println("Testing NN")
	for {
		count++
		test, err := reader2.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		for i := 1; i < 783; i++ {
			val, errStr := strconv.ParseFloat(test[i], 64)
			if errStr != nil {
				log.Fatalln("casting error", errStr)
				break
			}
			in[i-1][0] = (val / 255) * 0.9
		}
		val, errStr := strconv.Atoi(test[0])
		if errStr != nil {
			log.Fatalln("casting error", errStr)
		}

		var guess int = 0
		var max float64 = 0
		toCheck := NN.Query(in, NN1)
		for i := 0; i < len(toCheck); i++ {
			if toCheck[i][0] > max {
				guess = i
				max = toCheck[i][0]
			}
		}
		if guess == val {
			correct++
		}
		total++
		if total%1000 == 0 {
			fmt.Println(toCheck)
			fmt.Println(guess)
			fmt.Println(val)
		}
	}
	fmt.Println("####################")
	fmt.Println(correct)
	fmt.Println(total)
	fmt.Println(((correct / total) * 100))
	fmt.Println("####################")
	return &NN1
}

func main() {
	NN1 := *NN.NodeMaker(784, 200, 200, 200, 10)
	start := time.Now()
	for i := 0; i < 6; i++ {
		NN1 = *MNIST(NN1)
	}
	elapsed := time.Since(start)
	fmt.Println(elapsed)
}
