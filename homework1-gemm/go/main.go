package main
import(
	"fmt"
	"math/rand"
	"time"
)
const SIZE=1000
func main(){
	 A:=[SIZE*SIZE]float64{}
	 B:=[SIZE*SIZE]float64{}
	 C:=[SIZE*SIZE]float64{}
	rand.Seed(time.Now().Unix())

	for i:=0;i<SIZE;i++{
		A[i]=rand.Float64()
		B[i]=rand.Float64()
	}

	start:=time.Now().UnixNano()
    for mi := 0; mi < SIZE; mi++{
        for  ni := 0; ni < SIZE; ni++{
            for ki := 0; ki < SIZE; ki++{
                C[mi * SIZE + ni] += A[mi * SIZE + ki] * B[ki * SIZE + ni]
			}
        }
    }
	fmt.Print(float64(time.Now().UnixNano()-start)*1e-9)
	fmt.Print("\n")
	C=[SIZE*SIZE]float64{}
	var calc=func (C *[SIZE*SIZE]float64 ,mi int ,ni int  ){
		for ki := 0; ki < SIZE; ki++{
			C[mi * SIZE + ni] += A[mi * SIZE + ki] * B[ki * SIZE + ni]
		}
		
	}
	start2:=time.Now().UnixNano()
	for mi := 0; mi < SIZE; mi++{
        for  ni := 0; ni < SIZE; ni++{
            go calc(&C,mi,ni)
        }
    }
	fmt.Print(float64(time.Now().UnixNano()-start2)*1e-9)
}