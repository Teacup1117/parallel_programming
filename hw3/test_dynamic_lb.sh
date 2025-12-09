#!/bin/bash

echo "========================================"
echo "動態任務佇列負載平衡測試"
echo "Dynamic Work Queue Load Balancing Test"
echo "========================================"
echo ""

# Test parameters
CAM_X=-0.522
CAM_Y=2.874
CAM_Z=1.340
TAR_X=0
TAR_Y=0
TAR_Z=0

# Compile
echo "[Step 1] 編譯程式..."
make clean && make
if [ $? -ne 0 ]; then
    echo "❌ 編譯失敗！"
    exit 1
fi
echo "✅ 編譯成功"
echo ""

# Create output directory
mkdir -p test_outputs

echo "[Step 2] 測試不同解析度..."
echo "========================================"

# Test different resolutions
for SIZE in 256 512 800; do
    WIDTH=$SIZE
    HEIGHT=$SIZE
    
    echo ""
    echo "📊 解析度: ${WIDTH}x${HEIGHT}"
    echo "----------------------------------------"
    
    # Static mode
    echo -n "  [靜態模式] "
    OUTPUT_STATIC="test_outputs/static_${SIZE}x${SIZE}.png"
    ./hw3 $CAM_X $CAM_Y $CAM_Z $TAR_X $TAR_Y $TAR_Z $WIDTH $HEIGHT $OUTPUT_STATIC 2>&1 | grep "Render time"
    
    # Dynamic mode
    echo -n "  [動態模式] "
    OUTPUT_DYNAMIC="test_outputs/dynamic_${SIZE}x${SIZE}.png"
    ENABLE_LB=1 ./hw3 $CAM_X $CAM_Y $CAM_Z $TAR_X $TAR_Y $TAR_Z $WIDTH $HEIGHT $OUTPUT_DYNAMIC 2>&1 | grep "Render time"
    
    # Verify images are identical
    if diff -q $OUTPUT_STATIC $OUTPUT_DYNAMIC > /dev/null 2>&1; then
        echo "  ✅ 圖片驗證: 完全相同"
    else
        echo "  ⚠️  圖片驗證: 有差異 (可能是正常的浮點誤差)"
    fi
done

echo ""
echo "========================================"
echo "[Step 3] 測試不同場景"
echo "========================================"

# Test Scene 1: Simple (more uniform)
echo ""
echo "📸 場景 1: 簡單場景 (較均勻負載)"
echo "----------------------------------------"
SIMPLE_CAM="0 3 0"
echo -n "  [靜態] "
./hw3 $SIMPLE_CAM $TAR_X $TAR_Y $TAR_Z 512 512 test_outputs/scene1_static.png 2>&1 | grep "Render time"
echo -n "  [動態] "
ENABLE_LB=1 ./hw3 $SIMPLE_CAM $TAR_X $TAR_Y $TAR_Z 512 512 test_outputs/scene1_dynamic.png 2>&1 | grep "Render time"

# Test Scene 2: Complex (high imbalance)
echo ""
echo "📸 場景 2: 複雜場景 (高度不均負載)"
echo "----------------------------------------"
echo -n "  [靜態] "
./hw3 $CAM_X $CAM_Y $CAM_Z $TAR_X $TAR_Y $TAR_Z 512 512 test_outputs/scene2_static.png 2>&1 | grep "Render time"
echo -n "  [動態] "
ENABLE_LB=1 ./hw3 $CAM_X $CAM_Y $CAM_Z $TAR_X $TAR_Y $TAR_Z 512 512 test_outputs/scene2_dynamic.png 2>&1 | grep "Render time"

# Test Scene 3: Far view (extreme imbalance - lots of sky)
echo ""
echo "📸 場景 3: 遠距離視角 (極度不均 - 大量天空)"
echo "----------------------------------------"
FAR_CAM="-5 5 5"
echo -n "  [靜態] "
./hw3 $FAR_CAM $TAR_X $TAR_Y $TAR_Z 512 512 test_outputs/scene3_static.png 2>&1 | grep "Render time"
echo -n "  [動態] "
ENABLE_LB=1 ./hw3 $FAR_CAM $TAR_X $TAR_Y $TAR_Z 512 512 test_outputs/scene3_dynamic.png 2>&1 | grep "Render time"

echo ""
echo "========================================"
echo "測試完成！"
echo "========================================"
echo ""
echo "📁 輸出檔案位於: test_outputs/"
echo ""
echo "📊 結果分析："
echo "  - 如果動態模式在大部分測試中更快 → 負載平衡有效 ✅"
echo "  - 如果只在特定場景更快 → 場景相關，需要選擇性啟用 ⚠️"
echo "  - 如果全部都更慢 → 場景太均勻或圖片太小，不需要負載平衡 ❌"
echo ""
echo "💡 使用建議："
echo "  - 均勻場景: 使用靜態模式 (預設)"
echo "  - 不均勻場景: export ENABLE_LB=1"
echo ""
