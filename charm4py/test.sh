for i in {1..11}
do
	python3 -m charmrun.start +p4 task_bench.py  -steps 4 -width $((2 ** i)) -type stencil_1d
done

