@echo off






@REM CPU Config

del /q "profiling\profile_output\*"

build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 128
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 256
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 512
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 1024

cd profiling
set EXTENSION=cpu

cargo r
cd ..

@REM End CPU Config





@REM Naive Config

del /q "profiling\profile_output\*"

build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 128
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 256
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 512
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 1024

cd profiling
set EXTENSION=naive

cargo r
cd ..

@REM End Naive Config






@REM Work Efficient Config

del /q "profiling\profile_output\*"

build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 128
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 256
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 512
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 1024

cd profiling
set EXTENSION=work_efficient

cargo r
cd ..

@REM End Work Efficient Config







@REM Thread Efficient Config

del /q "profiling\profile_output\*"

build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 128
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 256
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 512
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 1024

cd profiling
set EXTENSION=thread_efficient

cargo r
cd ..

@REM End Thread Efficient Config













@REM Thrust Config

del /q "profiling\profile_output\*"

build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 128
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 256
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 512
build\bin\Release\cis5650_stream_compaction_test.exe -blocksize 1024

cd profiling
set EXTENSION=thrust

cargo r
cd ..

@REM End Thrust Config