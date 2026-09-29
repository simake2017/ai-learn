package main

import (
	"fmt"
	"sync"
)

// 这段代码有 5 个 bug，让 code-review 找出来

type Counter struct {
	mu    sync.Mutex
	count int
}

func (c *Counter) Inc() {
	c.count++ // bug 1: 没加锁
}

func (c *Counter) Get() int {
	return c.count // bug 2: 读也没加锁
}

func processItems(items []string) {
	var wg sync.WaitGroup
	for _, item := range items {
		wg.Add(1)
		go func() { // bug 3: 闭包捕获循环变量
			fmt.Println("processing:", item)
			wg.Done()
		}()
	}
	wg.Wait()
}

func readFile(path string) string {
	// bug 4: 没处理错误
	f, _ := os.Open(path)
	defer f.Close()
	buf := make([]byte, 1024)
	n, _ := f.Read(buf)
	return string(buf[:n])
}

func main() {
	m := make(map[string]int)
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			m["key"] = i // bug 5: 并发写 map 没加锁
			wg.Done()
		}(i)
	}
	wg.Wait()
}
