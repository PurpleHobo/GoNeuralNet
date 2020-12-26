package NN

import (
	"math"
	"math/rand"
)

type Nodes struct {
	w1, w2, w3, w4 [][]float64
}

func Matrix(row int, col int) [][]float64 {
	toRet := make([][]float64, row)
	for i := 0; i < row; i++ {
		toRet[i] = make([]float64, col)
		for j := 0; j < col; j++ {
			toRet[i][j] = rand.Float64() - 0.5
		}
	}
	return toRet
}

func MatDot(arr1 [][]float64, arr2 [][]float64) [][]float64 {
	var count float64
	toRet := make([][]float64, len(arr1))
	if len(arr1[0]) != len(arr2) {
		return toRet
	}
	for i := 0; i < len(arr1); i++ {
		toRet[i] = make([]float64, len(arr2[0]))
	}
	for col := 0; col < len(arr2[0]); col++ {
		for row := 0; row < len(arr1); row++ {
			count = 0
			for width := 0; width < len(arr1[0]); width++ {
				count += arr1[row][width] * arr2[width][col]
			}
			toRet[row][col] += count
		}
	}
	return toRet
}

func ReLU(arr [][]float64) [][]float64 {
	for i := 0; i < len(arr); i++ {
		for j := 0; j < len(arr[0]); j++ {
			arr[i][j] = math.Max(0, arr[i][j])
		}
	}
	return arr
}

func MatMulti(arr [][]float64, multi float64) [][]float64 {
	for i := 0; i < len(arr); i++ {
		for j := 0; j < len(arr[0]); j++ {
			arr[i][j] *= multi
		}
	}
	return arr
}

func MatPos(arr [][]float64) [][]float64 {
	for i := 0; i < len(arr); i++ {
		for j := 0; j < len(arr[0]); j++ {
			if arr[i][j] > 0 {
				arr[i][j] = 1
			} else {
				arr[i][j] = 0
			}
		}
	}
	return arr
}

func Query(arr [][]float64, NN Nodes) [][]float64 {
	arr = ReLU(MatDot(NN.w1, arr))
	arr = ReLU(MatDot(NN.w2, arr))
	arr = ReLU(MatDot(NN.w3, arr))
	return ReLU(MatDot(NN.w4, arr))
}

func Trans(arr [][]float64) [][]float64 {
	toRet := make([][]float64, len(arr[0]))
	for i := 0; i < len(arr[0]); i++ {
		toRet[i] = make([]float64, len(arr))
	}
	for col := 0; col < len(arr[0]); col++ {
		for row := 0; row < len(arr); row++ {
			toRet[col][row] = arr[row][col]
		}
	}
	return toRet
}

func MatEle(arr [][]float64, arr2 [][]float64, isMultiplication bool) [][]float64 {
	for col := 0; col < len(arr[0]); col++ {
		for row := 0; row < len(arr); row++ {
			if isMultiplication {
				arr[row][col] *= arr2[row][col]
			} else {
				arr[row][col] += arr2[row][col]
			}
		}
	}
	return arr
}

func DeExtreme(arr [][]float64) [][]float64 {
	for i := 0; i < len(arr); i++ {
		for j := 0; j < len(arr[0]); j++ {
			if arr[i][j] > 100000 || arr[i][j] < -100000 {
				arr[i][j] = rand.Float64() - 0.5
			}
		}
	}
	return arr
}

func Safety(NN Nodes) *Nodes {
	NN.w1 = DeExtreme(NN.w1)
	NN.w2 = DeExtreme(NN.w2)
	NN.w3 = DeExtreme(NN.w3)
	NN.w4 = DeExtreme(NN.w4)
	return &NN
}

func Train(input [][]float64, target [][]float64, lr float64, NN Nodes) *Nodes {
	var retNN Nodes
	var h1, h2, h3, h4, e1, e2, e3 [][]float64
	NN = *Safety(NN)
	retNN = NN
	h1 = ReLU(MatDot(NN.w1, input))
	h2 = ReLU(MatDot(NN.w2, h1))
	h3 = ReLU(MatDot(NN.w3, h2))
	h4 = ReLU(MatDot(NN.w4, h3))

	Error := make([][]float64, len(target))
	for i := 0; i < len(target); i++ {
		Error[i] = make([]float64, 1)
	}

	for i := 0; i < len(target); i++ {
		Error[i][0] = target[i][0] - h4[i][0]
	}
	e3 = MatDot(Trans(NN.w4), Error)
	e2 = MatDot(Trans(NN.w3), e3)
	e1 = MatDot(Trans(NN.w2), e2)

	retNN.w4 = MatEle(retNN.w4, MatMulti(MatDot(MatEle(Error, MatPos(h4), true), Trans(h3)), lr), false)
	retNN.w3 = MatEle(retNN.w3, MatMulti(MatDot(MatEle(e3, MatPos(h3), true), Trans(h2)), lr), false)
	retNN.w2 = MatEle(retNN.w2, MatMulti(MatDot(MatEle(e2, MatPos(h2), true), Trans(h1)), lr), false)
	retNN.w1 = MatEle(retNN.w1, MatMulti(MatDot(MatEle(e1, MatPos(h1), true), Trans(input)), lr), false)

	return &retNN
}

func MakeSimple(pos int, length int) [][]float64 {
	toRet := make([][]float64, length)
	for i := 0; i < length; i++ {
		toRet[i] = make([]float64, 1)
		toRet[i][0] = 1
	}
	toRet[pos][0] = 1000
	return toRet
}

func NodeMaker(input int, inner1 int, inner2 int, inner3 int, output int) *Nodes {
	toRet := Nodes{}
	toRet.w4 = Matrix(output, inner3)
	toRet.w3 = Matrix(inner3, inner2)
	toRet.w2 = Matrix(inner2, inner1)
	toRet.w1 = Matrix(inner1, input)
	return &toRet
}
