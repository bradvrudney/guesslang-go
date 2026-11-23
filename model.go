package guesslang

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

//go:embed model/model.bin
var modelBin []byte

//go:embed model/languages.json
var languagesJSON []byte

// tensor is one array of the model, row by row.
type tensor struct {
	dims []int
	data []float32
}

// network is guesslang's trained model: a deep part (the mean of the
// embeddings of a text's tokens, through two ReLU layers) and a wide part
// (the sum of per-token weights), added together and turned into
// probabilities by a softmax.
type network struct {
	embed      tensor // buckets x 70
	h0w, h0b   tensor // 70 x 512, 512
	h1w, h1b   tensor // 512 x 32, 32
	outw, outb tensor // 32 x classes, classes
	linw, linb tensor // buckets x classes, classes
	languages  []string
}

var (
	loadOnce sync.Once
	loaded   *network
	loadErr  error
)

func model() (*network, error) {
	loadOnce.Do(func() { loaded, loadErr = loadNetwork(modelBin, languagesJSON) })
	return loaded, loadErr
}

// languageOrder reads languages.json (name -> file extension) in the order
// the keys are written, which is the order of the model's outputs.
func languageOrder(raw []byte) ([]string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return nil, errors.New("languages.json is not an object")
	}
	var names []string
	for dec.More() {
		k, err := dec.Token()
		if err != nil {
			return nil, err
		}
		var ext string
		if err := dec.Decode(&ext); err != nil {
			return nil, err
		}
		names = append(names, k.(string))
	}
	return names, nil
}

func readTensors(b []byte) (map[string]tensor, error) {
	if len(b) < 6 || string(b[:4]) != "GLG1" {
		return nil, errors.New("not a guesslang model")
	}
	n := int(binary.LittleEndian.Uint16(b[4:]))
	b = b[6:]
	out := map[string]tensor{}
	short := errors.New("the model is cut short")
	for i := 0; i < n; i++ {
		if len(b) < 2 {
			return nil, short
		}
		nl := int(binary.LittleEndian.Uint16(b))
		if len(b) < 2+nl+1 {
			return nil, short
		}
		name := string(b[2 : 2+nl])
		nd := int(b[2+nl])
		b = b[3+nl:]
		if len(b) < 4*nd {
			return nil, short
		}
		dims, size := make([]int, nd), 1
		for d := range dims {
			dims[d] = int(binary.LittleEndian.Uint32(b[4*d:]))
			size *= dims[d]
		}
		b = b[4*nd:]
		if len(b) < 4*size {
			return nil, short
		}
		data := make([]float32, size)
		for k := range data {
			data[k] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*k:]))
		}
		b = b[4*size:]
		out[name] = tensor{dims, data}
	}
	return out, nil
}

func loadNetwork(bin, langs []byte) (*network, error) {
	ts, err := readTensors(bin)
	if err != nil {
		return nil, err
	}
	names, err := languageOrder(langs)
	if err != nil {
		return nil, err
	}
	c := len(names)
	nw := &network{languages: names}
	for _, f := range []struct {
		dst  *tensor
		name string
		dims []int
	}{
		{&nw.embed, "dnn/input_from_feature_columns/input_layer/content_embedding/embedding_weights", []int{buckets, 70}},
		{&nw.h0w, "dnn/hiddenlayer_0/kernel", []int{70, 512}},
		{&nw.h0b, "dnn/hiddenlayer_0/bias", []int{512}},
		{&nw.h1w, "dnn/hiddenlayer_1/kernel", []int{512, 32}},
		{&nw.h1b, "dnn/hiddenlayer_1/bias", []int{32}},
		{&nw.outw, "dnn/logits/kernel", []int{32, c}},
		{&nw.outb, "dnn/logits/bias", []int{c}},
		{&nw.linw, "linear/linear_model/content/weights", []int{buckets, c}},
		{&nw.linb, "linear/linear_model/bias_weights", []int{c}},
	} {
		t, ok := ts[f.name]
		if !ok {
			return nil, fmt.Errorf("the model lacks %s", f.name)
		}
		if fmt.Sprint(t.dims) != fmt.Sprint(f.dims) {
			return nil, fmt.Errorf("%s has shape %v, want %v", f.name, t.dims, f.dims)
		}
		*f.dst = t
	}
	return nw, nil
}

// dense computes relu?(in x w + b) for a w of len(in) x len(b).
func dense(in []float32, w, b tensor, relu bool) []float32 {
	out := make([]float32, len(b.data))
	copy(out, b.data)
	n := len(out)
	for i, x := range in {
		if x == 0 {
			continue
		}
		row := w.data[i*n : (i+1)*n]
		for j, wv := range row {
			out[j] += x * wv
		}
	}
	if relu {
		for j, v := range out {
			if v < 0 {
				out[j] = 0
			}
		}
	}
	return out
}

// probabilities runs the network on the token ids of a text.
func (nw *network) probabilities(ids []int) []float64 {
	c := len(nw.languages)
	emb := make([]float32, 70)
	lin := make([]float32, c)
	copy(lin, nw.linb.data)
	for _, id := range ids {
		row := nw.embed.data[id*70 : (id+1)*70]
		for j, v := range row {
			emb[j] += v
		}
		for j, v := range nw.linw.data[id*c : (id+1)*c] {
			lin[j] += v
		}
	}
	if len(ids) > 0 {
		inv := 1 / float32(len(ids))
		for j := range emb {
			emb[j] *= inv
		}
	}
	h := dense(emb, nw.h0w, nw.h0b, true)
	h = dense(h, nw.h1w, nw.h1b, true)
	logits := dense(h, nw.outw, nw.outb, false)
	probs := make([]float64, c)
	maxv := math.Inf(-1)
	for j := range logits {
		logits[j] += lin[j]
		maxv = math.Max(maxv, float64(logits[j]))
	}
	var sum float64
	for j, l := range logits {
		probs[j] = math.Exp(float64(l) - maxv)
		sum += probs[j]
	}
	for j := range probs {
		probs[j] /= sum
	}
	return probs
}

// Score is the probability that a text is written in a language.
type Score struct {
	Language    string
	Probability float64
}

// Probabilities gives the probability of each of the 54 languages the model
// knows for text, the most probable first.
func Probabilities(text string) ([]Score, error) {
	nw, err := model()
	if err != nil {
		return nil, err
	}
	probs := nw.probabilities(bucketsOf(text))
	out := make([]Score, len(probs))
	for i, p := range probs {
		out[i] = Score{nw.languages[i], p}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Probability > out[j].Probability })
	return out, nil
}
