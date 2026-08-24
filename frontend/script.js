const game = new Chess();

const board = Chessboard("board",{

    draggable:true,

    position:"start",

    pieceTheme:
    "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/{piece}.png",

    onDrop:onDrop
});

function onDrop(source,target){

    const move = game.move({

        from:source,

        to:target,

        promotion:"q"

    });

    if(move===null)
        return "snapback";

    updateHistory();

    // later call python backend here
}

function updateHistory(){

    document.getElementById("history").innerHTML =
    game.history().join("<br>");
}

document.getElementById("flip").onclick=()=>board.flip();

document.getElementById("newGame").onclick=()=>{

    game.reset();

    board.start();

    updateHistory();
};