$('.context-button').click(function() {
    var data_attr = $( this ).attr( "data-paraId" ).split("_");

    if ( data_attr[1] === "previous-paragraph") {
        $( $( "#para-" + data_attr[0] ).find( " div." + data_attr[1] + ":hidden").get().reverse() )
            .first()
            .attr("hidden", false);

    } else {
        $( "#para-" + data_attr[0] ).find( " div." + data_attr[1] + ":hidden")
            .first()
            .attr("hidden", false);
    }

    $( this ).siblings(".remove-all-context").attr( "hidden", false );
});

$(".remove-all-context").click( function() {
    var mainPara = $( this ).closest( ".main-paragraph" );

    $( $( mainPara ).find( ".previous-paragraph" ) ).attr( "hidden", true );
    $( $( mainPara ).find( ".following-paragraph" ) ).attr( "hidden", true );
    $( this ).attr( "hidden", true );

});

$('td').click(function() {
    var placement = $( this ).attr( "data-placement" );
    $( this ).closest( "." + placement ).attr("hidden", true);
});