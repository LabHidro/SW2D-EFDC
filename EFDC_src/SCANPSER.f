      SUBROUTINE SCANPSER  
      USE GLOBAL  
      OPEN(1,FILE='PSER.INP',STATUS='OLD')  
      DO NS=1,NPSER  
   10   READ(1,*,ERR=10,END=40)M,R,R,R,R  
        NDPSER=MAX(NDPSER,M)  
        DO I=1,M  
          READ(1,*,ERR=20,END=40)R,R  
        ENDDO  
      ENDDO  
      CLOSE(1)  
      RETURN  
C  
   20 WRITE(*,30)  
      WRITE(8,30)  
   30 FORMAT('READ ERROR IN INPUT FILE PSER.INP')
      STOP  
   40 WRITE(*,50)  
      WRITE(8,50)  
   50 FORMAT('UNEXPECTED END OF INPUT FILE PSER.INP')
      STOP  
      END  

