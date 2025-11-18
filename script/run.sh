nohup python /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 5 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt --test-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt  >  /home/ubuntu/clk/infinigen/log/flexgen.log &

#infinigen
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/infinigen/flex_opt.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/infinigen/pytorch_backend.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
nohup python /home/ubuntu/clk/infinigen/speedup/flexgen/infinigen/flex_opt.py --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 5 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt --test-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt  --alpha 4 --partial-weight-ratio 0.2 --max-num-kv 400 >  /home/ubuntu/clk/infinigen/log/infinigen.log &

#h2o
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/h2o/flex_opt.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/h2o/pytorch_backend.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
nohup python /home/ubuntu/clk/infinigen/speedup/flexgen/h2o/flex_opt.py --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 5 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt --test-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt  --max-num-kv 415 --hh-ratio 0.1 --hh-all >  /home/ubuntu/clk/infinigen/log/h2o.log &

#int4
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/original/flex_opt.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/original/pytorch_backend.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
nohup python /home/ubuntu/clk/infinigen/speedup/flexgen/original/flex_opt.py --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 5 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt --test-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt  --compress-cache >  /home/ubuntu/clk/infinigen/log/int4.log &

#spattn
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
rm /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/spattn/flex_opt.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/flex_opt.py
ln -s /home/ubuntu/clk/infinigen/speedup/flexgen/spattn/pytorch_backend.py /home/ubuntu/clk/infinigen/speedup/flexgen/flexgen/pytorch_backend.py
nohup /home/ubuntu/clk/infinigen/speedup/flexgen/spattn/flex_opt.py --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 5 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt --test-input-path /home/ubuntu/clk/infinigen/speedup/scripts/figure14/pg19_firstbook.txt  --compress-cache >  /home/ubuntu/clk/infinigen/log/int4.log &
